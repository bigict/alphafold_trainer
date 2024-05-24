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

"""Core modules, which have been refactored in AlphaFold-Multimer.

The main difference is that MSA sampling pipeline is moved inside the JAX model
for easier implementation of recycling and ensembling.

Lower-level modules up to EvoformerIteration are reused from modules.py.
"""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphafold.common import residue_constants
from alphafold.model import folding_multimer
from alphafold.model import modules
from alphafold.model import modules_multimer
from alphafold.model import prng
from alphafold.model import utils



class AlphaFoldIteration(hk.Module):
  """A single recycling iteration of AlphaFold architecture.

  Computes ensembled (averaged) representations from the provided features.
  These representations are then passed to the various heads
  that have been requested by the configuration file.
  """

  def __init__(self, config, global_config, name='alphafold_iteration'):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def __call__(self,
               batch,
               is_training,
               compute_loss=False,
               return_representations=False,
               safe_key=None):

    if is_training:
      num_ensemble = np.asarray(self.config.num_ensemble_train)
    else:
      num_ensemble = np.asarray(self.config.num_ensemble_eval)

    # Compute representations for each MSA sample and average.
    embedding_module = modules_multimer.EmbeddingsAndEvoformer(
        self.config.embeddings_and_evoformer, self.global_config)
    repr_shape = hk.eval_shape(
        lambda: embedding_module(batch, is_training))
    representations = {
        k: jnp.zeros(v.shape, v.dtype) for (k, v) in repr_shape.items()
    }

    def ensemble_body(x, unused_y):
      """Add into representations ensemble."""
      del unused_y
      representations, safe_key = x
      safe_key, safe_subkey = safe_key.split()
      representations_update = embedding_module(
          batch, is_training, safe_key=safe_subkey)

      for k in representations:
        if k not in {'msa', 'true_msa', 'bert_mask'}:
          representations[k] += representations_update[k] * (
              1. / num_ensemble).astype(representations[k].dtype)
        else:
          representations[k] = representations_update[k]

      return (representations, safe_key), None

    (representations, _), _ = hk.scan(
        ensemble_body, (representations, safe_key), None, length=num_ensemble)

    self.representations = representations
    self.batch = batch
    self.heads = {}
    for head_name, head_config in sorted(self.config.heads.items()):
      if not head_config.weight:
        continue  # Do not instantiate zero-weight heads.

      head_factory = {
          'masked_msa':
              modules.MaskedMsaHead,
          'distogram':
              modules.DistogramHead,
          'structure_module':
              folding_multimer.StructureModule,
          'predicted_aligned_error':
              modules.PredictedAlignedErrorHead,
          'predicted_lddt':
              modules.PredictedLDDTHead,
          'experimentally_resolved':
              modules.ExperimentallyResolvedHead,
          'logit_diff':
              modules.LogitDiffPathogenicityHead,
      }[head_name]
      self.heads[head_name] = (head_config,
                               head_factory(head_config, self.global_config))

    structure_module_output = None
    if 'entity_id' in batch and 'all_atom_positions' in batch:
      _, fold_module = self.heads['structure_module']
      structure_module_output = fold_module(representations, batch, is_training)

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

    for name, (head_config, module) in self.heads.items():
      if name == 'structure_module' and structure_module_output is not None:
        ret[name] = structure_module_output
        representations['structure_module'] = structure_module_output.pop('act')
      # Skip confidence heads until StructureModule is executed.
      elif name in {'predicted_lddt', 'predicted_aligned_error',
                    'experimentally_resolved'}:
        continue
      else:
        ret[name] = module(representations, batch, is_training)
      if compute_loss:
        total_loss += loss(module, head_config, ret, name)

    # Add confidence heads after StructureModule is executed.
    if self.config.heads.get('predicted_lddt.weight', 0.0):
      name = 'predicted_lddt'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch, is_training)

    if self.config.heads.experimentally_resolved.weight:
      name = 'experimentally_resolved'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch, is_training)
      if compute_loss:
        total_loss += loss(module, head_config, ret, name, filter_ret=False)

    if self.config.heads.get('predicted_aligned_error.weight', 0.0):
      name = 'predicted_aligned_error'
      head_config, module = self.heads[name]
      ret[name] = module(representations, batch, is_training)
      # Will be used for ipTM computation.
      ret[name]['asym_id'] = batch['asym_id']
      if compute_loss:
        total_loss += loss(module, head_config, ret, name, filter_ret=False)

    if compute_loss:
      return ret, total_loss
    else:
      return ret


class AlphaFold(hk.Module):
  """AlphaFold-Multimer model with recycling.
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
      return_representations=False,
      safe_key=None):

    c = self.config
    impl = AlphaFoldIteration(c, self.global_config)

    if safe_key is None:
      safe_key = prng.SafeKey(hk.next_rng_key())
    elif isinstance(safe_key, jnp.ndarray):
      safe_key = prng.SafeKey(safe_key)

    assert isinstance(batch, dict)
    num_res = batch['aatype'].shape[0]

    def get_prev(ret):
      new_prev = {
          'prev_pos':
              ret['structure_module']['final_atom_positions'],
          'prev_msa_first_row': ret['representations']['msa_first_row'],
          'prev_pair': ret['representations']['pair'],
      }
      return jax.tree_map(jax.lax.stop_gradient, new_prev)

    def apply_network(prev, safe_key):
      recycled_batch = {**batch, **prev}
      return impl(
          batch=recycled_batch,
          is_training=is_training,
          compute_loss=compute_loss,
          safe_key=safe_key)

    prev = {}
    emb_config = self.config.embeddings_and_evoformer
    if emb_config.recycle_pos:
      prev['prev_pos'] = jnp.zeros(
          [num_res, residue_constants.atom_type_num, 3])
    if emb_config.recycle_features:
      prev['prev_msa_first_row'] = jnp.zeros(
          [num_res, emb_config.msa_channel])
      prev['prev_pair'] = jnp.zeros(
          [num_res, num_res, emb_config.pair_channel])

    if self.config.num_recycle:
      if 'num_iter_recycling' in batch:
        # Training time: num_iter_recycling is in batch.
        # Value for each ensemble batch is the same, so arbitrarily taking 0-th.
        num_iter = batch['num_iter_recycling'][0]

        # Add insurance that even when ensembling, we will not run more
        # recyclings than the model is configured to run.
        num_iter = jnp.minimum(num_iter, c.num_recycle)
      else:
        # Eval mode or tests: use the maximum number of iterations.
        num_iter = c.num_recycle

      def distances(points):
        """Compute all pairwise distances for a set of points."""
        return jnp.sqrt(jnp.sum((points[:, None] - points[None, :])**2,
                                axis=-1))

      def recycle_body(x):
        i, _, prev, safe_key = x
        safe_key1, safe_key2 = safe_key.split() if c.resample_msa_in_recycling else safe_key.duplicate()  # pylint: disable=line-too-long
        ret = apply_network(prev=prev, safe_key=safe_key2)
        return i+1, prev, get_prev(ret), safe_key1

      def recycle_cond(x):
        i, prev, next_in, _ = x
        ca_idx = residue_constants.atom_order['CA']
        sq_diff = jnp.square(distances(prev['prev_pos'][:, ca_idx, :]) -
                             distances(next_in['prev_pos'][:, ca_idx, :]))
        mask = batch['seq_mask'][:, None] * batch['seq_mask'][None, :]
        sq_diff = utils.mask_mean(mask, sq_diff)
        # Early stopping criteria based on criteria used in
        # AF2Complex: https://www.nature.com/articles/s41467-022-29394-2
        diff = jnp.sqrt(sq_diff + 1e-8)  # avoid bad numerics giving negatives
        less_than_max_recycles = (i < num_iter)
        has_exceeded_tolerance = (
            (i == 0) | (diff > c.recycle_early_stop_tolerance))
        return less_than_max_recycles & has_exceeded_tolerance

      if hk.running_init():
        num_recycles, _, prev, safe_key = recycle_body(
            (0, prev, prev, safe_key))
      else:
        num_recycles, _, prev, safe_key = hk.while_loop(
            recycle_cond,
            recycle_body,
            (0, prev, prev, safe_key))
    else:
      # No recycling.
      num_recycles = 0

    # Run extra iteration.
    ret = apply_network(prev=prev, safe_key=safe_key)
    if compute_loss:
      ret = ret[0], [ret[1]]

    if not return_representations:
      del ret['representations']
    ret['num_recycles'] = num_recycles

    return ret
