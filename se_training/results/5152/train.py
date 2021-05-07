#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with spectral masking.

To run this recipe, do the following:
> python train.py train.yaml --data_folder /path/to/save/mini_librispeech

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Szu-Wei Fu 2020
 * Chien-Feng Liao 2020
 * Peter Plantinga 2021
"""
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from data_preparing.data_pipeline import dataio_prep
from se_brain import SEBrain

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments
    hparams_file = "train.yaml"

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
    )

    # Create dataset objects "train" and "valid"
    datasets = dataio_prep(hparams['data_path'])

    # Initialize the Brain object to prepare for mask training.
    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=hparams["run_opts"],
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load best checkpoint (highest STOI) for evaluation
    test_stats = se_brain.evaluate(
        test_set=datasets["test"],
        max_key="stoi",
        test_loader_kwargs=hparams["dataloader_options"],
    )
