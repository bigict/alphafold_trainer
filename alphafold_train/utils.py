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
from typing import Any

from jax.example_libraries import optimizers as jopt
import jax.numpy as jnp
import jax.random as jrand

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


def split_np_random_seed(rng: jrand.PRNGKey):
  rng, sub_rng = jrand.split(rng, 2)
  return rng, int(jrand.randint(sub_rng, [1], 0, INT_MAX).item(0))


def load_opt_state_from_pkl(pkl_path: str):
  with open(pkl_path, "rb") as f:
    params = pickle.load(f)
  opt_state = jopt.pack_optimizer_state(params)
  return opt_state


def load_params_from_npz(npz_path: str):
  params = jnp.load(npz_path, allow_pickle=True)
  return params["arr_0"].flat[0]


def load_params(model_path: str):
  if model_path.endswith(".pkl"):
    opt_state = load_opt_state_from_pkl(model_path)
    params = jopt.unpack_optimizer_state(opt_state)
  elif model_path.endswith(".npz"):
    params = load_params_from_npz(model_path)
  else:
    raise ValueError(f"unknown type of params: {model_path}")
  return params
