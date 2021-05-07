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


# configure data path
train_metadata_filepath = "D:\LibriMix\data\Libri1Mix\wav8k\min\metadata\mixture_libri2mix_train-100_mix_single.csv"
dev_metadata_filepath = "D:\LibriMix\data\Libri1Mix\wav8k\min\metadata\mixture_libri2mix_dev_mix_single.csv"
test_metadata_filepath = "D:\LibriMix\data\Libri1Mix\wav8k\min\metadata\mixture_libri2mix_test_mix_single.csv"
metadata_path = {'train': train_metadata_filepath, 'dev': dev_metadata_filepath, 'test': test_metadata_filepath}
datasets = dataio_prep(metadata_path)

# load pretrained model
run_opts = {"device": "cuda:0" if torch.cuda.is_available() else "cpu"}
asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech",
                                           savedir="./pretrained_ASR", run_opts=run_opts)

# load speech enhancement module
sys.path.append('D:\\ASRProj\\se_training')  # Add speech enhancement package to path
se_module = load_se_module("D:\\ASRProj\\se_training\\train.yaml")


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ..'
                     '、.%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


test_metadata = pd.read_csv(test_metadata_filepath)
errors = []
total_edits = []
total_scored_tokens = []
####################  test on entire test set ####################
for row in test_metadata.iterrows():
    # for each test sample
    progress(row[0], test_metadata.shape[0])
    test_sample = row[1]
    # load noisy wav, the dimension is 1(channel size) x sample size
    enhanced_wav, sample_rate = torchaudio.load(test_sample['mixture_path'])
    # preprocess with speech enhancement module
    # the module requires data to be dimension batch size x sample size. Here we inteprete the 1(channel size) as the batch size
    # therefore, the noisy_wav has dimension 1(channel size/batch size) x sample size
    # enhanced_wav = se_module(noisy_wav)
    # then pass to ASR after normalization, where the normalizer required samples size first, channel size second
    enhanced_wav = asr_model.audio_normalizer(enhanced_wav.transpose(0, 1), sample_rate)
    batch, rel_length = enhanced_wav.unsqueeze(0), torch.tensor([1.0])
    predicted_words, _ = asr_model.transcribe_batch(batch, rel_length)
    # compare results with ground truth
    hypothesis = predicted_words[0]
    # record each individual wer, number of edits, number of total scored words
    individual_wer, (edits, scored_tokens) = wer(test_sample['transcript'].split(), hypothesis.split())
    errors.append(individual_wer)
    total_edits.append(edits)
    total_scored_tokens.append(scored_tokens)

print(sum(errors) / len(errors))
print(sum(total_edits) / sum(total_scored_tokens) * 100)

############ generate some enhanced audio ##############
# num_samples = 10  # number of enhanced samples to generate
# samples = random.sample([i for i in range(len(test_metadata))], num_samples)
# output_dir = "enhanced_result"  # save the (noisy audio, enhanced audio) pair at output dir
# os.mkdir(output_dir)
# for s in samples:
#     test_sample = test_metadata.iloc[s]
#     noisy_wav, sample_rate = torchaudio.load(test_sample['mixture_path'])
#     enhanced_wav = se_module(noisy_wav)
#     # save to dir
#     dir_path = os.path.join(output_dir, str(s))
#     os.mkdir(dir_path)
#     torchaudio.save(os.path.join(dir_path, "noisy.wav"), noisy_wav, 8000)
#     torchaudio.save(os.path.join(dir_path, "enhanced.wav"), enhanced_wav.cpu(), 8000)


