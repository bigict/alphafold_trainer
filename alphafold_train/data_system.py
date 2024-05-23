# Copyright 2021 Beijing DP Technology Co., Ltd.
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
"""Data system used to load training datasets."""

import os
import contextlib
import functools
import glob
import json
import multiprocessing as mp
import pickle
from typing import Optional, Union

from absl import logging
import jax
import jax.random as jrand
import numpy as np
from Bio.PDB import protein_letters_3to1

from alphafold.common import residue_constants as rc
from alphafold.data import mmcif_parsing
from alphafold.data.templates import _get_atom_positions as get_atom_positions
from alphafold.model.features import FeatureDict
from alphafold.model.features import np_example_to_features as process_features
from alphafold_train import utils
from alphafold_train.data.pipeline import process_labels

FEATNAME_DICT = set([
    "aatype", "residue_index", "seq_length", "template_aatype",
    "template_all_atom_masks", "template_all_atom_positions",
    "template_sum_probs", "is_distillation", "seq_mask", "msa_mask",
    "msa_row_mask", "random_crop_to_size_seed", "template_mask",
    "template_pseudo_beta", "template_pseudo_beta_mask", "atom14_atom_exists",
    "residx_atom14_to_atom37", "residx_atom37_to_atom14", "atom37_atom_exists",
    "extra_msa", "extra_msa_mask", "extra_msa_row_mask", "bert_mask",
    "true_msa", "extra_has_deletion", "extra_deletion_value", "msa_feat",
    "target_feat"
])


def _pid_split(prot_name):
  # assert naming styles are in `101m_A` or `101m_1_A`
  i, j = prot_name.find("_"), prot_name.rfind("_")
  pid = prot_name[:i] if i != -1 else prot_name
  chain = prot_name[j + 1:] if j != -1 else None
  return pid, chain


def cif_to_fasta(mmcif_object: mmcif_parsing.MmcifObject, chain_id: str) -> str:
  residues = mmcif_object.seqres_to_structure[chain_id]
  residue_names = [residues[t].name for t in range(len(residues))]
  residue_letters = [protein_letters_3to1.get(n, "X") for n in residue_names]
  filter_out_triple_letters = lambda x: x if len(x) == 1 else "X"
  fasta_string = "".join(
      [filter_out_triple_letters(n) for n in residue_letters])
  return fasta_string


ignored_keys = [
  "domain_name",
  "sequence",
  "template_domain_names",
  "template_e_value",
  "template_neff",
  "template_prob_true",
  "template_release_date",
  "template_score",
  "template_similarity",
  "template_sequence",
  "template_sum_probs"
]

batched_keys = [
  "deletion_matrix_int",
  "msa",
  "template_aatype",
  "template_all_atom_masks",
  "template_all_atom_positions",
  "template_confidence_scores"
]


def crop_and_pad(raw_features: FeatureDict,
                 raw_labels: FeatureDict,
                 random_seed: int,
                 crop_size: int = 256,
                 pad_for_shorter_seq: bool = True) -> FeatureDict:

  # keys that should be ignored when conducting crop & pad
  def is_ignored_key(k):
    return k in ignored_keys

  # keys that have batch dim, e.g. msa features which have shape
  # [N_msa, N_res, ...]
  def is_batched_key(k):
    return k in batched_keys

  # get num res from aatype
  assert "aatype" in raw_features.keys(), \
      "'aatype' missing from batch, which is not expected."
  num_res = raw_features["aatype"].shape[0]

  if num_res < crop_size and pad_for_shorter_seq:
    # pad short seq (0 padding and (automatically) create masks)
    def pad(key: str, array: np.ndarray):
      if is_ignored_key(key):
        return array
      d_seq = 1 if is_batched_key(key) else 0  # choose the dim to crop / pad
      pad_shape = list(array.shape)
      pad_shape[d_seq] = crop_size - num_res
      pad_array = np.zeros(pad_shape)
      pad_array = pad_array.astype(array.dtype)
      array = np.concatenate([array, pad_array], axis=d_seq)
      return array

    raw_features = {k: pad(k, v) for k, v in raw_features.items()}
    raw_labels = {k: pad(k, v) for k, v in raw_labels.items()}
  elif num_res > crop_size:
    # crop long seq.
    np.random.seed(random_seed)
    crop_start = np.random.randint(num_res - crop_size)
    crop_end = crop_start + crop_size

    def crop(key: str, array: np.ndarray):
      if is_ignored_key(key):
        return array
      if is_batched_key(key):
        return array[:, crop_start:crop_end, ...]
      else:
        return array[crop_start:crop_end, ...]

    raw_features = {k: crop(k, v) for k, v in raw_features.items()}
    raw_labels = {k: crop(k, v) for k, v in raw_labels.items()}
  else:
    # seq len == crop size
    pass

  # fix for input seq length
  raw_features["seq_length"] = (
      crop_size * np.ones_like(raw_features["seq_length"])).astype(np.int32)

  return raw_features, raw_labels


def remove_masked_residues(raw_labels: FeatureDict):
  mask = raw_labels["all_atom_mask"][:, 0].astype(bool)
  return {k: v[mask] for k, v in raw_labels.items()}


class DataSystem:
  """Analog to pytorch Dataset"""
  def __init__(
      self,
      model_config,  # model config.
      data_config):  # data config mainly including paths.
    # copy config
    self.mc = model_config
    self.dc = data_config

    # get crop size
    self.crop_size = self.mc.data.eval.crop_size

    # get sample_weights
    if self.dc.sample_weights is None:
      sample_weights = DataSystem.get_sample_weights_from_dir(
          self.dc.features_dir)
    else:  # use all entries under
      sample_weights = DataSystem.get_sample_weights_from_json(
          self.dc.sample_weights)

    self.prot_keys = list(sample_weights.keys())
    # unify the sample weights as sample probability
    sum_weights = sum(sample_weights.values())
    self.sample_prob = [v / sum_weights for v in sample_weights.values()]
    self.num_prot = len(self.prot_keys)
    self.check_completeness()

  def check_completeness(self):
    # check that every protein has mmcif as labels.
    self.pdb_list = DataSystem.get_pdb_list_from_dir(self.dc.mmcif_dir)
    for prot_name in self.prot_keys:
      pdb_id, _ = _pid_split(prot_name)
      assert pdb_id in self.pdb_list, (
          f"{prot_name} doesn't have the corresponding mmcif file in"
          f"{self.dc.mmcif_dir}.")
    logging.debug("checking for data completeness successful.")

  def load(self, prot_name: str):
    # load msa, template etc from features.pkl
    features_path = os.path.join(self.dc.features_dir,
                                 prot_name,
                                 "features.pkl")
    with open(features_path, "rb") as f:
      raw_features = pickle.load(f)

    # load labels from mmcif
    pdb_id, chain_id = _pid_split(prot_name)
    chain_id = utils.default(chain_id, "A")
    mmcif_string = utils.get_file_contents(
        os.path.join(self.dc.mmcif_dir, f"{pdb_id}.cif"))
    mmcif_obj = mmcif_parsing.parse(file_id=pdb_id,
                                    mmcif_string=mmcif_string).mmcif_object
    assert utils.exists(mmcif_obj), (prot_name,)
    all_atom_positions, all_atom_mask = get_atom_positions(
        mmcif_obj, chain_id, max_ca_ca_distance=float("inf"))
    # directly parses sequence from fasta. should be consistent to 'aatype' in
    # input features (from .fasta or .pkl)
    sequence = cif_to_fasta(mmcif_obj, chain_id)
    aatype_idx = np.array([rc.restype_order_with_x[rn] for rn in sequence])
    resolution = np.array([mmcif_obj.header["resolution"]])

    raw_labels = {
        "aatype_index":       aatype_idx,           # [NR,]
        "all_atom_positions": all_atom_positions,   # [NR, 37, 3]
        "all_atom_mask":      all_atom_mask,        # [NR, 37]
        "resolution":         resolution            # [,]
    }
    if hasattr(self.dc, "variant_dir"):
      pass
      # raw_labels.update(self.load_variants(
      #     self.dc.variant_dir, pdb_id=pdb_id, chain_id=chain_id))
    return raw_features, raw_labels


  def preprocess(
      self,
      rng: jrand.PRNGKey,
      raw_features: FeatureDict,
      raw_labels: FeatureDict) -> FeatureDict:
    rng, crop_seed = utils.split_np_random_seed(rng)
    raw_features, raw_labels = crop_and_pad(raw_features,
                                            raw_labels,
                                            crop_size=self.crop_size,
                                            pad_for_shorter_seq=True,
                                            random_seed=crop_seed)
    rng, feat_seed = utils.split_np_random_seed(rng)
    processed_features = process_features(raw_features,
                                          config=self.mc,
                                          random_seed=feat_seed)
    with jax.disable_jit():  # using jit here is experimentally slower
      processed_labels = process_labels(raw_labels)
    batch = {**processed_features, **processed_labels}
    return rng, batch

  def sample(self, rng: jrand.PRNGKey, batch_size: Optional[int] = None):
    """
    pick a (batch of) protein(s) randomly and generate rng(s) for processing.
    if batch_size is None, return a pair of result; otherwise return a list of
    pairs.
    """
    rng, seed = utils.split_np_random_seed(rng)
    np.random.seed(seed)
    if batch_size is None:
      prot_idx = np.random.choice(self.num_prot, p=self.sample_prob)
      return rng, prot_idx
    else:  # this code is not used.
      prot_idxs = np.random.choice(self.num_prot,
                                   size=batch_size,
                                   replace=(batch_size > self.num_prot),
                                   p=self.sample_prob)
      rngs = list(jrand.split(rng, batch_size))
      return list(zip(rngs, prot_idxs))

  def get_batch(self, prot_idx: int, rng: jrand.PRNGKey):
    prot_name = self.prot_keys[prot_idx % self.num_prot]
    logging.debug("loading protein #%06d: %s...", prot_idx, prot_name)
    raw_features, raw_labels = self.load(prot_name)
    resolution = raw_labels.pop("resolution")
    rng, batch = self.preprocess(rng, raw_features, raw_labels)
    batch["resolution"] = resolution
    rng, batch_rng = jrand.split(rng, 2)
    return rng, batch_rng, batch

  def random_recycling(self, step: int, batch: FeatureDict):
    """
    Generate the number of recycling iterations for a given step and add it to
    the batch. This method is specifically set here to make sure the result is
    equal among workers at each step.
    """
    rng = jrand.PRNGKey(step)
    num_iter_recycling = jrand.randint(rng, [1], 0,
                                       self.mc.model.num_recycle + 1)
    batch["num_iter_recycling"] = num_iter_recycling
    return batch

  def batch_gen(self, rng: jrand.PRNGKey):
    with jax.disable_jit():
      while True:
        rng, prot_idx = self.sample(rng, None)
        rng, batch_rng, batch = self.get_batch(prot_idx, rng)
        yield batch_rng, batch

  @staticmethod
  def get_sample_weights_from_dir(features_dir):
    sample_weights = {
        os.path.basename(p): 1
        for p in glob.glob(features_dir + "/*")
        if os.path.isdir(p)
    }
    assert len(list(sample_weights.keys())) > 0, \
        "no sub-directories under given feature directory %s." % (features_dir)
    return sample_weights

  @staticmethod
  def get_sample_weights_from_json(json_path):
    with open(json_path, "r") as f:
      sample_weights = json.load(f)
    if isinstance(sample_weights, list):
      sample_weights = {k: 1 for k in sample_weights}
    return sample_weights

  @staticmethod
  def get_pdb_list_from_dir(mmcif_dir):
    pdb_list = [
        os.path.basename(mmcif).replace(".cif", "")
        for mmcif in glob.glob(mmcif_dir + "/*.cif")
    ]
    return pdb_list


# macros for retrying getting queue items.
MAX_TIMEOUT = 60
MAX_FAILED = 5


class GetBatchProcess(mp.Process):
  """
  a multiprocessing worker to conduct data loading.
  remark: make sure no jax call is used before this worker starts,
          or the XLA-in-fork issue could arise. (notably, there are
          tensorflow calls in `DataSystem.preprocess()`. )
  """
  def __init__(
      self,
      data: DataSystem,
      num_batches: int,           # number of batches to generate
      is_training: bool = True,   # if true, random recycling is used.
      random_seed: int = 0,
      max_queue_size: int = 8,
      mpi_rank: Union[int, mp.Value] = 0,
      mpi_cond: mp.Condition = None):
    mp.Process.__init__(self)

    self.queue = mp.Queue(max_queue_size)
    self.data = data
    self.num_batches = num_batches
    self.is_training = is_training
    self.random_seed = random_seed
    self.mpi_rank = mpi_rank
    self.mpi_cond = mpi_cond

  def __next__(self) -> tuple[jrand.PRNGKey, FeatureDict]:
    # waiting time upperbound = MAX_FAILED * MAX_TIMEOUT
    for t in range(MAX_FAILED):
      try:
        item = self.queue.get(block=True, timeout=MAX_TIMEOUT)
        logging.debug("get queue item succeeded. current qsize = %s.",
                      self.queue.qsize())
        return item
      except:  # pylint: disable=bare-except
        logging.warning("get queue item timeout after %ss (%s/%s).", MAX_TIMEOUT,
                        t + 1, MAX_FAILED)
    # exit subprogram:
    logging.error("get queue item failed for too many times. subprogram quit.")
    return (None, None)

  def run(self):
    mpi_rank = self.mpi_rank
    if self.mpi_cond:
      with self.mpi_cond:
        while mpi_rank.value == -1:
          self.mpi_cond.wait()
      mpi_rank = mpi_rank.value

    logging.info("get batch process [%s] started.", mpi_rank)
    with jax.disable_jit():
      rng = jrand.PRNGKey(self.random_seed)
      rng = jrand.fold_in(rng, mpi_rank)
    batch_gen = self.data.batch_gen(rng)
    for step in range(self.num_batches):
      batch_rng, batch = next(batch_gen)
      if self.is_training:
        batch = self.data.random_recycling(step, batch)
      self.queue.put((batch_rng, batch))
      logging.info("write queue item %s. current qsize = %s.", step,
                   self.queue.qsize())
    logging.info("get batch process [%s] finished.", mpi_rank)


class GetBatchProcessMgr():
  """Analog to pytorch dataset worker"""
  def __init__(self, *args, **kwargs):
    self.batch_process = functools.partial(GetBatchProcess, *args, **kwargs)
    self.items = []

  def create(self, *args, **kwargs) -> GetBatchProcess:
    proc = self.batch_process(*args, **kwargs)
    self.items.append(proc)
    return proc

  def start(self):
    for proc in self.items:
      if not proc.is_alive():
        proc.start()

  def stop(self):
    for proc in self.items:
      logging.info("terminate process %s ...", proc.name)
      if proc.is_alive():
        proc.terminate()


@contextlib.contextmanager
def dataset_manager(*args, **kwargs) -> GetBatchProcessMgr:
  proc_mgr = GetBatchProcessMgr(*args, **kwargs)
  try:
    yield proc_mgr
  finally:
    proc_mgr.stop()
