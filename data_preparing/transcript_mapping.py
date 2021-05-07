"""
This file adds ground truth transcript to LibriMix metadata csv
"""
import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, default='D:\LibriMix\data\LibriSpeech',
                    help='Path to librispeech root directory')
parser.add_argument('--metadata_dir', type=str, default='D:\LibriMix\data\Libri1Mix\wav8k\min\metadata',
                    help='Path to the generated LibriMix dataset')

def main(args):
    # Get librispeech root path
    librispeech_dir = args.librispeech_dir
    # Get Metadata directory
    metadata_dir = args.metadata_dir
    # Get the metadata containing noisy audio + clean audio location
    metadata_files = [os.path.join(metadata_dir, f) for f in os.listdir(metadata_dir) if f.startswith('mixture')]
    # Append transcript to the metadata files, so each entry contains (noisy audio location, clean audio location, ground truth transcript)
    for f in metadata_files:
        print('Processing', f)
        # append_transcripts(librispeech_dir, f)
        rename_fields(f)

def append_transcripts(librispeech_dir: str, metadata_filename: str) -> None:
    # append the transcript to for each audio pair
    metadata = pd.read_csv(metadata_filename)
    if 'transcript' not in metadata.columns:
        # Append a transcript column at the end with default empty string
        metadata.insert(len(metadata.columns), 'transcript', '')
    # for each row, find its transcript in LirbiSpeech dataset
    for row_index, row in metadata.iterrows():
        mixture_id = row['mixture_ID']
        speaker_text_line = mixture_id.split('_')[0]
        metadata.loc[row_index, 'transcript'] = get_transcript(librispeech_dir, metadata_filename, speaker_text_line)
    # save to current csv
    metadata.to_csv(metadata_filename, index=False)

def get_transcript(librispeech_dir: str, metadata_filename: str, speaker_text_line: str) -> str:
    # read the transcript line at the corresponding speaker, text, and line number
    if metadata_filename.find('dev') != -1:
            transcript_dir = os.path.join(librispeech_dir, 'dev-clean')
    elif metadata_filename.find('test') != -1:
            transcript_dir = os.path.join(librispeech_dir, 'test-clean')
    elif metadata_filename.find('train') != -1:
            transcript_dir = os.path.join(librispeech_dir, 'train-clean-100')
    else:
        print('Cannot find corresponding LibriSpeech dir for', metadata_filename)
        raise FileNotFoundError

    speaker, text, line = speaker_text_line.split('-')
    transcript_filename = speaker + '-' + text + '.trans.txt'
    transcript_path = os.path.join(transcript_dir, speaker, text, transcript_filename)

    with open(transcript_path, 'r') as transcript:
        content = transcript.readlines()
        for l in content:
            if (i := l.find(speaker_text_line)) != -1:
                return l[i + len(speaker_text_line) + 1: -1]
    return ''

def rename_fields(metadata_filename: str) -> None:
    '''rename the column name according to speechbrain format'''
    metadata = pd.read_csv(metadata_filename)
    metadata = metadata.rename(columns={'mixture_ID': 'ID'})
    metadata.to_csv(metadata_filename, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)