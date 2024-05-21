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

"""Utilities for training AlphaFold."""

import os
import gzip
import pickle
from typing import *

from absl import logging
import numpy as np
from jax.example_libraries import optimizers as jopt
import jax.numpy as jnp
import jax.random as jrand

from alphafold.data.pipeline import FeatureDict

INT_MAX = 0x7fffffff

def exists(val: Any):
  return val is not None

def default(val: Any, d: Any):
  return val if exists(val) else d

def get_file_contents(pathname: str):
  if os.path.exists(pathname):
    with open(pathname, "r") as f:
      return f.read()
  else:
    with gzip.open(f"{pathname}.gz", "rt") as f:
      return f.read()
  raise ValueError(f"{pathname} not exist.")

ignored_keys = [
  'domain_name',
  'sequence',
  'template_domain_names',
  'template_e_value',
  'template_neff',
  'template_prob_true',
  'template_release_date',
  'template_score',
  'template_similarity',
  'template_sequence',
  'template_sum_probs'
]

batched_keys = [
  'deletion_matrix_int',
  'msa',
  'template_aatype',
  'template_all_atom_masks',
  'template_all_atom_positions',
  'template_confidence_scores'
]

def crop_and_pad(
    raw_features: FeatureDict,
    raw_labels: FeatureDict,
    random_seed: int,
    crop_size: int = 256,
    pad_for_shorter_seq: bool = True) -> FeatureDict:

  # keys that should be ignored when conducting crop & pad
  def is_ignored_key(k):
    return k in ignored_keys

  # keys that have batch dim, e.g. msa features which have shape [N_msa, N_res, ...]
  def is_batched_key(k):
    return k in batched_keys

  # get num res from aatype
  assert 'aatype' in raw_features.keys(), \
      "'aatype' missing from batch, which is not expected."
  num_res = raw_features['aatype'].shape[0]

  if num_res < crop_size and pad_for_shorter_seq:
    # pad short seq (0 padding and (automatically) create masks)
    def pad(key: str, array: np.ndarray):
      if is_ignored_key(key):
        return array
      d_seq = 1 if is_batched_key(key) else 0           # choose the dim to crop / pad
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
  raw_features['seq_length'] = (crop_size * np.ones_like(raw_features['seq_length'])).astype(np.int32)

  return raw_features, raw_labels

def remove_masked_residues(raw_labels: FeatureDict):
  mask = raw_labels['all_atom_mask'][:,0].astype(bool)
  return {k: v[mask] for k, v in raw_labels.items()}

def split_np_random_seed(rng):
  rng, sub_rng = jrand.split(rng, 2)
  return rng, int(jrand.randint(sub_rng, [1], 0, INT_MAX).item(0))


def load_opt_state_from_pkl(pkl_path: str):
    params = pickle.load(open(pkl_path, "rb"))
    opt_state = jopt.pack_optimizer_state(params)
    return opt_state


def load_params_from_npz(npz_path: str):
    params = jnp.load(npz_path, allow_pickle=True)
    return params['arr_0'].flat[0]


def load_params(model_path: str):
  if model_path.endswith('.pkl'):
    opt_state = load_opt_state_from_pkl(model_path)
    params = jopt.unpack_optimizer_state(opt_state)
  elif model_path.endswith('.npz'):
    params = load_params_from_npz(model_path)
  else:
    raise ValueError(f"unknown type of params: {model_path}")
  return params

