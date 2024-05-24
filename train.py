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
import multiprocessing as mp

from absl import app
from absl import flags
from absl import logging

# internal import
from alphafold.model.config import model_config as get_model_config
from alphafold_train.data_system import DataSystem, dataset_manager
from alphafold_train.train_config import train_config
from alphafold_train.trainer import Trainer

LOG_VERBOSITY = {
  "FATAL": logging.FATAL,
  "ERROR": logging.ERROR,
  "WARN": logging.WARNING,
  "WARNING": logging.WARNING,
  "INFO": logging.INFO,
  "DEBUG": logging.DEBUG
}
gc = train_config.global_config

flags.DEFINE_boolean("use_mpi", gc.use_mpi, "Use MPI to train with multiple "
                     "GPUs")
flags.DEFINE_enum(
    "model_name", gc.model_name, [
        "model_1", "model_2", "model_3", "model_4", "model_5",
        "model_1_ptm", "model_2_ptm", "model_3_ptm", "model_4_ptm", "model_5_ptm",  # pylint: disable=line-too-long
        "model_1_multimer_v3", "model_2_multimer_v3", "model_3_multimer_v3", "model_4_multimer_v3", "model_5_multimer_v3"  # pylint: disable=line-too-long
    ], "Choose preset model configuration - the monomer model, "
    "the monomer model with extra ensembling, monomer model with "
    "pTM head, or multimer model")
flags.DEFINE_enum(
    "verbose", gc.verbose.upper(), LOG_VERBOSITY.keys(), "Verbose")

FLAGS = flags.FLAGS


def main(argv):
  gc.use_mpi = FLAGS.use_mpi
  gc.model_name = FLAGS.model_name
  gc.verbose = FLAGS.verbose

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  logging.set_verbosity(LOG_VERBOSITY[gc.verbose.upper()])

  mp.set_start_method("spawn")

  #########################################
  # main function of training (single gpu).
  #########################################

  # get configs
  model_config = get_model_config(gc.model_name)
  model_config.update(train_config.model_config)
  logging.info(model_config)

  # construct datasets
  logging.info("constructing train data ...")
  train_data = DataSystem(gc, model_config, train_config.data.train)
  logging.info("constructing validation data ...")
  try:
    eval_data = DataSystem(gc, model_config, train_config.data.eval)
  except:  # pylint: disable=bare-except
    logging.warning(
        "failed to load validation data. poor configurations may be provided.")
    eval_data = None

  # NOTE: start all children before initialize MPI
  mpi_rank, mpi_cond = mp.Value("i", -1), mp.Condition()
  with dataset_manager(
      random_seed=gc.random_seed,
      max_queue_size=gc.max_queue_size,
      mpi_cond=mpi_cond,
      # pass rank to generate different batches among mpi.
      mpi_rank=mpi_rank) as mgr:
    # create batch processes
    train_data_proc = mgr.create(
        data=train_data,
        # add 1 for the initialization batch
        num_batches=gc.end_step - gc.start_step + 1,
        is_training=True)

    if eval_data is not None:
      eval_data_proc = mgr.create(
          data=eval_data,
          num_batches=(gc.end_step - gc.start_step) // gc.eval_freq + 1,
          is_training=False)
    mgr.start()

    if gc.use_mpi:
      from mpi4py import MPI  # pylint: disable=import-outside-toplevel
      mpi_comm = MPI.COMM_WORLD
      mpi_rank.value = mpi_comm.Get_rank()
      is_main_process = (mpi_rank.value == 0)
      os.environ["CUDA_VISIBLE_DEVICES"] = str(
          mpi_rank.value % gc.gpus_per_node)
    else:  # assume single gpu is used.
      mpi_comm = None
      mpi_rank.value = 0
      is_main_process = True
    with mpi_cond:
      mpi_cond.notify_all()

    # define and initialize trainer
    trainer = Trainer(global_config=gc,
                      model_config=model_config,
                      mpi_comm=mpi_comm)
    logging.info("initializing ...")
    # do NOT use the returned rng to initialize trainer.
    _, init_batch = next(train_data_proc)
    trainer.initialize(init_batch, load_format=gc.ckpt_format)

    # conduct training
    logging.info("training ...")
    for step in range(gc.start_step, gc.end_step):
      update_rng, batch = next(train_data_proc)
      trainer.train_step(step, batch, update_rng, silent=(not is_main_process))
      if eval_data is not None and trainer.is_eval_step(step):
        eval_rng, batch = next(eval_data_proc)
        trainer.eval_step(step, batch, eval_rng, silent=(not is_main_process))
    logging.info("finished training.")


if __name__ == "__main__":
  app.run(main)
