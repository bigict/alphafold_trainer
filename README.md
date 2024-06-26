# AlphaFold Trainer

## Motivation
Train [AlphaFold](http://github.com/deepmind/alphafold) related models with easy.

## FAQ
1. How to run it?
  * Clone the code.
  ```
  $ git clone git@github.com:bigict/alphafold_trainer.git
  $ cd alphafold_trainer
  $ git submodule update --init --recursive --remote
  ```
  * Install dependencies.
  ```
  $ conda create -n alphafold2 python=3.9
  $ conda activate alphafold2
  $ bash install_dependencies.sh
  ```
  * Run test.
  ```
  $ export PYTHONPATH=`pwd`/alphafold
  ```
  ```
  $ XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py
  ```
  or run with multi-gpus
  ```
  $ MPI4JAX_USE_CUDA_MPI=1 XLA_PYTHON_CLIENT_PREALLOCATE=false mpirun --mca opal_cuda_support 1 -n 1 python train.py --use_mpi
  ```
