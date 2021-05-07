import sys
import os
import torch
import torchaudio
import pandas as pd

from se_training.load_se_module import load_se_module

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

# configure data path
train_metadata_filepath = "D:\LibriMix\data\Libri1Mix\wav8k\min\metadata\mixture_libri2mix_train-100_mix_single.csv"
dev_metadata_filepath = "D:\LibriMix\data\Libri1Mix\wav8k\min\metadata\mixture_libri2mix_dev_mix_single.csv"
test_metadata_filepath = "D:\LibriMix\data\Libri1Mix\wav8k\min\metadata\mixture_libri2mix_test_mix_single.csv"

for metadata_filepath in [train_metadata_filepath]:
    metadata = pd.read_csv(metadata_filepath)
    # add a enhanced audio column in the metadata file
    if 'enhanced_path' not in metadata.columns:
        # Append a transcript column at the end with default empty string
        metadata.insert(len(metadata.columns), 'enhanced_path', '')
    # for each sample in the dataset:
    for row_index, sample in metadata.iterrows():
        progress(row_index, metadata.shape[0])
        mixture_id = sample['ID']
        #  generate its enhanced version
        noisy_wav, sample_rate = torchaudio.load(sample['mixture_path'])
        enhanced_wav = se_module(noisy_wav)
        #  save to enhanced folder using ID as name
        base_dir = sample['mixture_path'].rsplit('\\', 2)[0]  # remove \\mix_single\\file.wav suffix
        enhanced_file_path = os.path.join(base_dir, 'enhanced', mixture_id + '.wav')
        torchaudio.save(enhanced_file_path, enhanced_wav.cpu(), sample_rate)
        #  write the file location to metadata
        metadata.loc[row_index, 'enhanced_path'] = enhanced_file_path
    # save to current csv
    metadata.to_csv(metadata_filepath, index=False)
