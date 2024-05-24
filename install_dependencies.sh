#!/bin/bash
# 
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

# Shell script for installing the environment.
# Usage: bash install_dependencies.sh

#######################################
# dependencies of feature processing  #
#######################################

set -e

CUDA_DIR=${CUDA_DIR:-/usr/local/cuda}
CUDA_VERSION=12.2.2

# install conda packages
conda install -y -c nvidia/label/cuda-12.2.2 cuda cuda-toolkit cudnn
#conda install -y -c conda-forge openmm=8.0.0 pdbfixer
# conda install -y -c bioconda hmmer hhsuite kalign2
 
# # update openmm
# work_path=$(pwd)
# python_path=$(which python)
# cd $(dirname $(dirname $python_path))/lib/python3.8/site-packages
# patch -p0 < $work_path/openmm.patch
# cd $work_path
 
# #######################################
# # dependencies of training AlphaFold  #
# #######################################
# 
# install openmpi
conda install -y -c conda-forge mpi4py=3.1.4 openmpi=4.0.4
pip install mpi4jax

# wget -c https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
# gunzip -c openmpi-4.1.1.tar.gz | tar xf -
# pushd openmpi-4.1.1
# ./configure --prefix=${HOME}/.local --enable-mca-dso=btl-smcuda,rcache-rgpusm,rcache-gpusm,accelerator-cuda --with-cuda=${CUDA_DIR}
# make -j 16 all install
# popd


# install conda and pip packages
pip install --upgrade pip \
    && pip install -r ./requirements.txt \
    && pip install -U jax==0.4.26 jaxlib==0.4.26+cuda12.cudnn89 \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# download stereo_chemical_props.txt
wget -c -P alphafold/alphafold/common \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
