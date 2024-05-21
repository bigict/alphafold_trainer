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

"""Training AlphaFold protein structure prediction model."""

# OS & MPI config. please config before any import of jax / tf.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from alphafold_train.train_config import train_config

# external import
import multiprocessing as mp

from absl import logging

# internal import
from alphafold.model.config import model_config as get_model_config
from alphafold_train.data_system import DataSystem, dataset_manager
from alphafold_train.trainer import Trainer


def train(train_config):
  mp.set_start_method('spawn')

  """
  main function of training (single gpu).
  """
  # get configs
  gc = train_config.global_config
  model_config = get_model_config(gc.model_name)
  model_config.update(train_config.model_config)
  logging.info(model_config)

  # construct datasets
  logging.info('constructing train data ...')
  train_data = DataSystem(model_config, train_config.data.train)
  logging.info('constructing validation data ...')
  try:
    eval_data = DataSystem(model_config, train_config.data.eval)
  except:
    logging.warning('failed to load validation data. poor configurations may be provided.')
    eval_data = None

  mpi_rank, mpi_cond = mp.Value('i', -1), mp.Condition()
  with dataset_manager(random_seed=gc.random_seed,
                       max_queue_size=gc.max_queue_size,
                       mpi_cond=mpi_cond,             # pass rank to generate different batches among mpi.
                       mpi_rank=mpi_rank) as mgr:
    # create batch processes
    train_data_proc = mgr.create(
        data=train_data,
        num_batches=gc.end_step - gc.start_step + 1,  # add 1 for the initialization batch
        is_training=True)

    if eval_data is not None:
      eval_data_proc = mgr.create(
          data=eval_data,
          num_batches=(gc.end_step - gc.start_step) // gc.eval_freq + 1,
          is_training=False)
    mgr.start()

    use_mpi = train_config.global_config.use_mpi
    if use_mpi:
      from mpi4py import MPI
      mpi_comm = MPI.COMM_WORLD
      mpi_size = mpi_comm.Get_size()
      mpi_rank.value = mpi_comm.Get_rank()
      is_main_process = (mpi_rank.value == 0)
      os.environ['CUDA_VISIBLE_DEVICES'] = str(mpi_rank.value % train_config.global_config.gpus_per_node)
    else:         # assume single gpu is used.
      mpi_comm = None
      mpi_rank.value = 0
      is_main_process = True
    with mpi_cond:
      mpi_cond.notify_all()

    # define and initialize trainer
    trainer = Trainer(
        global_config=gc,
        optim_config=train_config.optimizer,
        model_config=model_config,
        mpi_comm=mpi_comm)
    logging.info('initializing ...')
    _, init_batch = next(train_data_proc)    # do NOT use the returned rng to initialize trainer.
    trainer.initialize(init_batch, load_format=gc.ckpt_format)

    # conduct training
    logging.info('training ...')
    for step in range(gc.start_step, gc.end_step):
      update_rng, batch = next(train_data_proc)
      trainer.train_step(step, batch, update_rng, silent=(not is_main_process))
      if eval_data is not None and trainer.is_eval_step(step):
        eval_rng, batch = next(eval_data_proc)
        trainer.eval_step(step, batch, eval_rng, silent=(not is_main_process))
    logging.info('finished training.')

  # if train_batch_proc.is_alive():
  #   train_batch_proc.terminate()
  # if eval_data is not None and eval_batch_proc.is_alive():
  #   eval_batch_proc.terminate()


if __name__ == '__main__':
  LOG_VERBOSITY = {
    'FATAL': logging.FATAL,
    'ERROR': logging.ERROR,
    'WARN': logging.WARNING,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG
  }
  # if is_main_process:
  logging.set_verbosity(LOG_VERBOSITY[train_config.global_config.verbose.upper()])
  # else:
  #   logging.set_verbosity(logging.ERROR)
  train(train_config)

