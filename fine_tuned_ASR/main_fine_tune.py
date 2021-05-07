import os
import random

import torch
import torchaudio
import sys
import pandas as pd
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR
from data_preparing.data_pipeline import dataio_prep
from wer import wer
from se_training.load_se_module import load_se_module
import speechbrain
from fine_tuned_ASR.EncDecFineTune import EncDecFineTune
from data_preparing.asr_pipeline import fine_tune_prep


# configure data path
train_metadata_filepath = "D:\LibriMix\data\Libri1Mix\wav8k\min\metadata\mixture_libri2mix_train-100_mix_single.csv"
dev_metadata_filepath = "D:\LibriMix\data\Libri1Mix\wav8k\min\metadata\mixture_libri2mix_dev_mix_single.csv"
test_metadata_filepath = "D:\LibriMix\data\Libri1Mix\wav8k\min\metadata\mixture_libri2mix_test_mix_single.csv"
metadata_path = {'train': train_metadata_filepath, 'dev': dev_metadata_filepath, 'test': test_metadata_filepath}
datasets = dataio_prep(metadata_path)
fine_tune_dataset = fine_tune_prep(metadata_path)

# load pretrained model
run_opts = {"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech",
                                           savedir="./pretrained_ASR", run_opts=run_opts)
modules = {"enc": asr_model.modules.asr_model[0],
           "emb": asr_model.modules.asr_model[1],
           "dec": asr_model.modules.asr_model[2],
           "compute_features": asr_model.modules.compute_features,  # we use the same features
           "normalize": asr_model.modules.normalize,
           "seq_lin": asr_model.modules.asr_model[-1],
           "beam_searcher": asr_model.modules.beam_searcher,
           }

num_epoches = 15
epoch_counter = speechbrain.utils.epoch_loop.EpochCounter(limit=num_epoches)
hparams = {"seq_cost": lambda x, y, z: speechbrain.nnet.losses.nll_loss(x, y, z, label_smoothing=0.1),
           "log_softmax": speechbrain.nnet.activations.Softmax(apply_log=True),
           "epoch_counter": epoch_counter,
           "checkpointer": speechbrain.utils.checkpoints.Checkpointer(checkpoints_dir='./fine_tuned_ASR',
                                                                      recoverables={'model': torch.nn.ModuleList([modules['enc'], modules['emb'], modules['dec'], modules['seq_lin']]),
                                                                                    'counter': epoch_counter}),
           "error_rate_computer": speechbrain.utils.metric_stats.ErrorRateStats,
           "train_logger": speechbrain.utils.train_logger.FileTrainLogger(save_file="./fine_tuned_ASR/train_log.txt"),
           "wer_file": "./fine_tuned_ASR/test_wer.txt"
           }


sb.create_experiment_directory(
    experiment_directory='./fine_tuned_ASR'
)

brain = EncDecFineTune(modules, hparams=hparams, opt_class=lambda x: torch.optim.Adam(x, 0.0001), run_opts=run_opts, checkpointer=hparams["checkpointer"])
brain.tokenizer = asr_model.tokenizer
brain.fit(epoch_counter, train_set=fine_tune_dataset['train'], valid_set=fine_tune_dataset['dev'],
          train_loader_kwargs={"batch_size": 8, "drop_last":True, "shuffle": False},
          valid_loader_kwargs={"batch_size": 8, "drop_last":True, "shuffle": False})
test_perf = brain.evaluate(
    test_set=fine_tune_dataset['test'],
    min_key="wer",
    test_loader_kwargs={"batch_size": 8, "drop_last":True, "shuffle": False},
)

# test_metadata = pd.read_csv(train_metadata_filepath)
# errors = []
# test_sample = test_metadata.iloc[2]
# predicted_words = asr_model.transcribe_file(test_sample['enhanced_path'])
# # compare results with ground truth
# hypothesis = predicted_words
# errors.append(wer(test_sample['transcript'].split(), hypothesis.split()))


