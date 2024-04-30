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

# CUDA=12.2.2
# JAX=0.4.26
# CUDNN=89

CUDA=11.8.0
JAX=0.4.25
CUDNN=86

# install conda packages
conda install -y -c nvidia cuda=${CUDA}
conda install -y -c conda-forge openmm=8.0.0 pdbfixer
conda install -y -c bioconda hmmer hhsuite kalign2
 
# # update openmm
# work_path=$(pwd)
# python_path=$(which python)
# cd $(dirname $(dirname $python_path))/lib/python3.8/site-packages
# patch -p0 < $work_path/openmm.patch
# cd $work_path
 
# #######################################
# # dependencies of training AlphaFold   #
# #######################################
# 
# install openmpi
conda install -y -c conda-forge mpi4py mpi4jax openmpi

# wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
# gunzip -c openmpi-4.1.1.tar.gz | tar xf -
# cd openmpi-4.1.1
# ./configure --prefix=${HOME}/.local
# make all install
# cd ..


# install conda and pip packages
# conda install -y -c nvidia cudnn==8.0.4
pip install --upgrade pip \
    && pip install -r ./requirements.txt \
    && pip install --upgrade jax==${JAX} jaxlib==${JAX}+cuda$(cut -f1 -d. <<< ${CUDA}).cudnn${CUDNN} -f \
    https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# download stereo_chemical_props.txt
wget -c -P alphafold/alphafold/common \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
