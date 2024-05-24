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
from typing import Any


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
