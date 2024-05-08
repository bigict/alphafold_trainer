# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules and code used in the core part of AlphaFold.

The structure generation code is in 'folding.py'.
"""
import functools
import math

import haiku as hk
import jax
import jax.numpy as jnp

from alphafold.common import residue_constants
from alphafold.model import common_modules
from alphafold.model import folding
from alphafold.model import modules


def clipped_sigmoid_cross_entropy(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    clip_negative_at_logit: float,
    clip_positive_at_logit: float,
    epsilon: float = 1e-07,
    ) -> jnp.ndarray:
  """Computes sigmoid xent loss with clipped input logits.

  Args:
    logits: The predicted values.
    labels: The ground truth values.
    clip_negative_at_logit: clip the loss to 0 if prediction smaller than this
      value for the negative class.
    clip_positive_at_logit: clip the loss to this value if prediction smaller
      than this value for the positive class.
    epsilon: A small increment to add to avoid taking a log of zero.

  Returns:
    Loss value.
  """
  prob = jax.nn.sigmoid(logits)
  prob = jnp.clip(prob, epsilon, 1. - epsilon)
  loss = -labels * jnp.log(
      prob) - (1. - labels) * jnp.log(1. - prob)
  loss_at_clip = math.log(math.exp(clip_negative_at_logit) + 1)
  loss = jnp.where(
      (1 - labels) * (logits < clip_negative_at_logit), loss_at_clip, loss)
  loss_at_clip = math.log(math.exp(-clip_positive_at_logit) + 1)
  loss = jnp.where(
      labels * (logits < clip_positive_at_logit), loss_at_clip, loss)
  return loss


class LogitDiffPathogenicityHead(hk.Module):
  """Variant pathogenicity classification head."""

  def __init__(self, config, global_config, name = 'logit_diff_head'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config
    self.num_output = len(residue_constants.restypes_with_x_and_gap)
    self.variant_row = 1

  def __call__(self,
               representations,
               batch,
               is_training):
    logits = common_modules.Linear(
        self.num_output,
        initializer='linear',
        name='logits')(
            representations['msa'][self.variant_row])

    ref_score = jnp.einsum('ij, ij->i', logits, jax.nn.one_hot(
        batch['aatype'], num_classes=self.num_output))
    variant_score = jnp.einsum('ij, ij->i', logits, jax.nn.one_hot(
        batch['variant_aatype'], num_classes=self.num_output))
    logit_diff = ref_score - variant_score
    variant_pathogenicity = jnp.sum(logit_diff * batch['variant_mask'])
    return {'variant_row_logit_diff': logit_diff,
            'variant_pathogenicity': variant_pathogenicity}

  def loss(self, value, batch):
    loss = clipped_sigmoid_cross_entropy(logits=value['variant_row_logit_diff'],
                                         labels=batch['pathogenicity'],
                                         clip_negative_at_logit=0.0,
                                         clip_positive_at_logit=-1.0)
    loss = (jnp.sum(loss * batch['variant_mask'], axis=(-2, -1)) /
            (1e-8 + jnp.sum(batch['variant_mask'], axis=(-2, -1))))
    return loss


class AlphaFoldIteration(hk.Module):
  """A single recycling iteration of AlphaFold architecture.

  Computes ensembled (averaged) representations from the provided features.
  These representations are then passed to the various heads
  that have been requested by the configuration file. Each head also returns a
  loss which is combined as a weighted sum to produce the total loss.

  Jumper et al. (2021) Suppl. Alg. 2 "Inference" lines 3-22
  """

  def __init__(self, config, global_config, name='alphafold_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               ensembled_batch,
               non_ensembled_batch,
               is_training,
               compute_loss=False,
               ensemble_representations=False,
               return_representations=False):

    num_ensemble = jnp.asarray(ensembled_batch['seq_length'].shape[0])

    if not ensemble_representations:
      assert ensembled_batch['seq_length'].shape[0] == 1

    def slice_batch(i):
      b = {k: v[i] for k, v in ensembled_batch.items()}
      b.update(non_ensembled_batch)
      return b

    # Compute representations for each batch element and average.
    evoformer_module = modules.EmbeddingsAndEvoformer(
        self.config.embeddings_and_evoformer, self.global_config)
    batch0 = slice_batch(0)
    representations = evoformer_module(batch0, is_training)

    # MSA representations are not ensembled so
    # we don't pass tensor into the loop.
    msa_representation = representations['msa']
    del representations['msa']

    # Average the representations (except MSA) over the batch dimension.
    if ensemble_representations:
      def body(x):
        """Add one element to the representations ensemble."""
        i, current_representations = x
        feats = slice_batch(i)
        representations_update = evoformer_module(
            feats, is_training)

        new_representations = {}
        for k in current_representations:
          new_representations[k] = (
              current_representations[k] + representations_update[k])
        return i+1, new_representations

      if hk.running_init():
        # When initializing the Haiku module, run one iteration of the
        # while_loop to initialize the Haiku modules used in `body`.
        _, representations = body((1, representations))
      else:
        _, representations = hk.while_loop(
            lambda x: x[0] < num_ensemble,
            body,
            (1, representations))

      for k in representations:
        if k != 'msa':
          representations[k] /= num_ensemble.astype(representations[k].dtype)

    representations['msa'] = msa_representation
    batch = batch0  # We are not ensembled from here on.

    heads = {}
    for head_name, head_config in sorted(self.config.heads.items()):
      if not head_config.weight:
        continue  # Do not instantiate zero-weight heads.

      head_factory = {
          'masked_msa': modules.MaskedMsaHead,
          'distogram': modules.DistogramHead,
          'structure_module': functools.partial(
              folding.StructureModule, compute_loss=compute_loss),
          'predicted_lddt': modules.PredictedLDDTHead,
          'predicted_aligned_error': modules.PredictedAlignedErrorHead,
          'experimentally_resolved': modules.ExperimentallyResolvedHead,
          'logit_diff': LogitDiffPathogenicityHead,
      }[head_name]
      heads[head_name] = (head_config,
                          head_factory(head_config, self.global_config))

    total_loss = 0.
    ret = {}
    ret['representations'] = representations

    def loss(module, head_config, ret, name, filter_ret=True):
      if filter_ret:
        value = ret[name]
      else:
        value = ret
      loss_output = module.loss(value, batch)
      ret[name].update(loss_output)
      loss = head_config.weight * ret[name]['loss']
      return loss

    for name, (head_config, module) in heads.items():
      # Skip PredictedLDDTHead and PredictedAlignedErrorHead until
      # StructureModule is executed.
      if name in ('predicted_lddt', 'predicted_aligned_error'):
        continue
      else:
        ret[name] = module(representations, batch, is_training)
        if 'representations' in ret[name]:
          # Extra representations from the head. Used by the structure module
          # to provide activations for the PredictedLDDTHead.
          representations.update(ret[name].pop('representations'))
      if compute_loss:
        total_loss += loss(module, head_config, ret, name)

    if self.config.heads.get('predicted_lddt.weight', 0.0):
      # Add PredictedLDDTHead after StructureModule executes.
      name = 'predicted_lddt'
      # Feed all previous results to give access to structure_module result.
      head_config, module = heads[name]
      ret[name] = module(representations, batch, is_training)
      if compute_loss:
        total_loss += loss(module, head_config, ret, name, filter_ret=False)

    if ('predicted_aligned_error' in self.config.heads
        and self.config.heads.get('predicted_aligned_error.weight', 0.0)):
      # Add PredictedAlignedErrorHead after StructureModule executes.
      name = 'predicted_aligned_error'
      # Feed all previous results to give access to structure_module result.
      head_config, module = heads[name]
      ret[name] = module(representations, batch, is_training)
      if compute_loss:
        total_loss += loss(module, head_config, ret, name, filter_ret=False)

    if compute_loss:
      return ret, total_loss
    else:
      return ret


class AlphaFold(hk.Module):
  """AlphaFold model with recycling.

  Jumper et al. (2021) Suppl. Alg. 2 "Inference"
  """

  def __init__(self, config, name='alphafold'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config

  def __call__(
      self,
      batch,
      is_training,
      compute_loss=False,
      ensemble_representations=False,
      return_representations=False):
    """Run the AlphaFold model.

    Arguments:
      batch: Dictionary with inputs to the AlphaFold model.
      is_training: Whether the system is in training or inference mode.
      compute_loss: Whether to compute losses (requires extra features
        to be present in the batch and knowing the true structure).
      ensemble_representations: Whether to use ensembling of representations.
      return_representations: Whether to also return the intermediate
        representations.

    Returns:
      When compute_loss is True:
        a tuple of loss and output of AlphaFoldIteration.
      When compute_loss is False:
        just output of AlphaFoldIteration.

      The output of AlphaFoldIteration is a nested dictionary containing
      predictions from the various heads.
    """

    impl = AlphaFoldIteration(self.config, self.global_config)
    batch_size, num_residues = batch['aatype'].shape

    def get_prev(ret):
      new_prev = {
          'prev_pos':
              ret['structure_module']['final_atom_positions'],
          'prev_msa_first_row': ret['representations']['msa_first_row'],
          'prev_pair': ret['representations']['pair'],
      }
      return jax.tree_map(jax.lax.stop_gradient, new_prev)

    def do_call(prev,
                recycle_idx,
                compute_loss=compute_loss):
      if self.config.resample_msa_in_recycling:
        num_ensemble = batch_size // (self.config.num_recycle + 1)
        def slice_recycle_idx(x):
          start = recycle_idx * num_ensemble
          size = num_ensemble
          return jax.lax.dynamic_slice_in_dim(x, start, size, axis=0)
        ensembled_batch = jax.tree_map(slice_recycle_idx, batch)
      else:
        num_ensemble = batch_size
        ensembled_batch = batch

      non_ensembled_batch = jax.tree_map(lambda x: x, prev)

      return impl(
          ensembled_batch=ensembled_batch,
          non_ensembled_batch=non_ensembled_batch,
          is_training=is_training,
          compute_loss=compute_loss,
          ensemble_representations=ensemble_representations)

    prev = {}
    emb_config = self.config.embeddings_and_evoformer
    if emb_config.recycle_pos:
      prev['prev_pos'] = jnp.zeros(
          [num_residues, residue_constants.atom_type_num, 3])
    if emb_config.recycle_features:
      prev['prev_msa_first_row'] = jnp.zeros(
          [num_residues, emb_config.msa_channel])
      prev['prev_pair'] = jnp.zeros(
          [num_residues, num_residues, emb_config.pair_channel])

    if self.config.num_recycle:
      if 'num_iter_recycling' in batch:
        # Training time: num_iter_recycling is in batch.
        # The value for each ensemble batch is the same, so arbitrarily taking
        # 0-th.
        num_iter = batch['num_iter_recycling'][0]

        # Add insurance that we will not run more
        # recyclings than the model is configured to run.
        num_iter = jnp.minimum(num_iter, self.config.num_recycle)
      else:
        # Eval mode or tests: use the maximum number of iterations.
        num_iter = self.config.num_recycle

      body = lambda x: (x[0] + 1,  # pylint: disable=g-long-lambda
                        get_prev(do_call(x[1], recycle_idx=x[0],
                                         compute_loss=False)))
      if hk.running_init():
        # When initializing the Haiku module, run one iteration of the
        # while_loop to initialize the Haiku modules used in `body`.
        _, prev = body((0, prev))
      else:
        _, prev = hk.while_loop(
            lambda x: x[0] < num_iter,
            body,
            (0, prev))
    else:
      num_iter = 0

    ret = do_call(prev=prev, recycle_idx=num_iter)
    if compute_loss:
      ret = ret[0], [ret[1]]

    if not return_representations:
      del (ret[0] if compute_loss else ret)['representations']  # pytype: disable=unsupported-operands
    return ret
