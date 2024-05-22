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
from typing import Union

from absl import logging
import jax
import jax.numpy as jnp
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

FEATNAME_DICT = set(['aatype', 'residue_index', 'seq_length', 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 'template_sum_probs', 'is_distillation', 'seq_mask', 'msa_mask', 'msa_row_mask', 'random_crop_to_size_seed', 'template_mask', 'template_pseudo_beta', 'template_pseudo_beta_mask', 'atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'extra_msa', 'extra_msa_mask', 'extra_msa_row_mask', 'bert_mask', 'true_msa', 'extra_has_deletion', 'extra_deletion_value', 'msa_feat', 'target_feat'])


def _pid_split(prot_name):
  # assert naming styles are in `101m_A` or `101m_1_A`
  i, j = prot_name.find("_"), prot_name.rfind("_")
  pid = prot_name[:i] if i != -1 else prot_name
  chain = prot_name[j + 1:] if j != -1 else None
  return pid, chain


def cif_to_fasta(mmcif_object: mmcif_parsing.MmcifObject,
                 chain_id: str) -> str:
  residues = mmcif_object.seqres_to_structure[chain_id]
  residue_names = [residues[t].name for t in range(len(residues))]
  residue_letters = [protein_letters_3to1.get(n, 'X') for n in residue_names]
  filter_out_triple_letters = lambda x: x if len(x) == 1 else 'X'
  fasta_string = ''.join([filter_out_triple_letters(n) for n in residue_letters])
  return fasta_string


class DataSystem:
  def __init__(self,
               model_config,      # model config.
               data_config):      # data config mainly including paths.
    # copy config
    self.mc = model_config
    self.dc = data_config

    # get crop size
    self.crop_size = self.mc.data.eval.crop_size

    # get sample_weights
    if self.dc.sample_weights is None:
      sample_weights = DataSystem.get_sample_weights_from_dir(self.dc.features_dir)
    else:   # use all entries under
      sample_weights = DataSystem.get_sample_weights_from_json(self.dc.sample_weights)

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
      assert pdb_id in self.pdb_list, \
          "%s doesn't have the corresponding mmcif file in %s." % (prot_name, self.dc.mmcif_dir)
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
    mmcif_string = utils.get_file_contents(os.path.join(self.dc.mmcif_dir,
                                                        f"{pdb_id}.cif"))
    mmcif_obj = mmcif_parsing.parse(file_id=pdb_id,
                                    mmcif_string=mmcif_string).mmcif_object
    assert utils.exists(mmcif_obj), (prot_name,)
    all_atom_positions, all_atom_mask = get_atom_positions(
        mmcif_obj, chain_id, max_ca_ca_distance=float("inf"))
    # directly parses sequence from fasta. should be consistent to 'aatype' in
    # input features (from .fasta or .pkl)
    sequence = cif_to_fasta(mmcif_obj, chain_id)
    aatype_idx = np.array(
        [rc.restype_order_with_x[rn] for rn in sequence])
    resolution = np.array(
        [mmcif_obj.header['resolution']])
    
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
      rng,
      raw_features: FeatureDict,
      raw_labels: FeatureDict) -> FeatureDict:
    rng, crop_seed = utils.split_np_random_seed(rng)
    raw_features, raw_labels = utils.crop_and_pad(
        raw_features, raw_labels,
        crop_size=self.crop_size,
        pad_for_shorter_seq=True,
        random_seed=crop_seed)
    rng, feat_seed = utils.split_np_random_seed(rng)
    processed_features = process_features(
        raw_features,
        config=self.mc,
        random_seed=feat_seed)
    with jax.disable_jit():           # using jit here is experimentally slower
      processed_labels = process_labels(raw_labels)
    batch = {**processed_features, **processed_labels}
    return rng, batch


  def sample(
      self,
      rng,
      batch_size = None):
    """
    pick a (batch of) protein(s) randomly and generate rng(s) for processing.
    if batch_size is None, return a pair of result; otherwise return a list of pairs.
    """
    rng, seed = utils.split_np_random_seed(rng)
    np.random.seed(seed)
    if batch_size is None:
      prot_idx = np.random.choice(self.num_prot, p=self.sample_prob)
      return rng, prot_idx
    else:         # this code is not used.
      prot_idxs = np.random.choice(
          self.num_prot,
          size=batch_size,
          replace=(batch_size > self.num_prot),
          p=self.sample_prob)
      rngs = list(jrand.split(rng, batch_size))
      return list(zip(rngs, prot_idxs))
  
  def get_batch(
      self,
      prot_idx,
      rng):
    prot_name = self.prot_keys[prot_idx % self.num_prot]
    logging.debug(f"loading protein #{prot_idx:06d}: {prot_name}...")
    raw_features, raw_labels = self.load(prot_name)
    resolution = raw_labels.pop('resolution')
    rng, batch = self.preprocess(rng, raw_features, raw_labels)
    batch['resolution'] = resolution
    rng, batch_rng = jrand.split(rng, 2)
    return rng, batch_rng, batch
  

  def random_recycling(
      self,
      step,
      batch):
    """
    generate the number of recycling iterations for a given step and add it to the batch.
    this method is specifically set here to make sure the result is equal among workers at each step.
    """
    rng = jrand.PRNGKey(step)
    num_iter_recycling = jrand.randint(rng, [1], 0, self.mc.model.num_recycle + 1)
    batch['num_iter_recycling'] = num_iter_recycling
    return batch

  
  def batch_gen(self, rng):
    with jax.disable_jit():
      while True:
        rng, prot_idx = self.sample(rng, None)
        rng, batch_rng, batch = self.get_batch(prot_idx, rng)
        yield batch_rng, batch


  @staticmethod
  def get_sample_weights_from_dir(features_dir):
    sample_weights = {
      os.path.basename(p): 1
          for p in glob.glob(features_dir + "/*") if os.path.isdir(p)
    }
    assert len(list(sample_weights.keys())) > 0, \
        "no sub-directories under given feature directory %s." % (features_dir)
    return sample_weights


  @staticmethod
  def get_sample_weights_from_json(json_path):
    try:
      sample_weights = json.load(open(json_path, 'r'))
    except:
      raise ValueError("failed to load sample weights from json file %s." % json_path)
    if isinstance(sample_weights, list):
      sample_weights = {k: 1 for k in sample_weights}
    return sample_weights


  @staticmethod
  def get_pdb_list_from_dir(mmcif_dir):
    pdb_list = [
        os.path.basename(mmcif).replace('.cif','')
            for mmcif in glob.glob(mmcif_dir + "/*.cif")]
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

  def __next__(self):
    # waiting time upperbound = MAX_FAILED * MAX_TIMEOUT
    for t in range(MAX_FAILED):
      try:
        item = self.queue.get(block=True, timeout=MAX_TIMEOUT)
        logging.debug("get queue item succeeded. current qsize = %s.",
                      self.queue.qsize())
        return item
      except:
        logging.warning(f"get queue item timeout after {MAX_TIMEOUT}s "
                        f"({t + 1}/{MAX_FAILED}).")
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
      logging.info(f"write queue item %s. current qsize = %s.",
                   step, self.queue.qsize())
    logging.info("get batch process [%s] finished.", mpi_rank)
    return

class GetBatchProcessMgr():
  def __init__(self, *args, **kwargs):
    self.batch_process = functools.partial(GetBatchProcess, *args, **kwargs)
    self.items = []

  def create(self, *args, **kwargs):
    proc = self.batch_process(*args, **kwargs)
    self.items.append(proc)
    return proc

  def start(self):
    for proc in self.items:
      if not proc.is_alive():
        proc.start()

  def stop(self):
    for proc in self.items:
      logging.info('terminate process %s ...', proc.name)
      if proc.is_alive():
        proc.terminate()

@contextlib.contextmanager
def dataset_manager(*args, **kwargs):
  proc_mgr = GetBatchProcessMgr(*args, **kwargs)
  try:
    yield proc_mgr
  finally:
    proc_mgr.stop()
