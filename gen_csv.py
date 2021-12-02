import os
import csv

import librosa
import numpy as np

def gen(data_dir: str, output_filename: str, include_genre: bool):
    header = "filename ZeroCrossingRate SpectralCentroid SpectralRolloff"
    
    for i in range(1, 21):
            header += f' MFCC{i}'
    for i in range(1, 13):
        header += f' ChrFrq{i}'

    if include_genre:
        header += ' genre'

    header = header.split()

    file = open(output_filename, 'w', newline='')
    writer = csv.writer(file)

    writer.writerow(header)

    for filename in os.listdir(data_dir):
        # Find the file name and song number
        num_idx = 0
        while not filename[num_idx].isdigit():
            num_idx += 1
        end_idx = num_idx
        while filename[end_idx].isdigit():
            end_idx += 1
        genre = filename[: num_idx]
        song_num = filename[num_idx : end_idx + 1]

        # Load the song and get sampling rate
        song, sr = librosa.load(f'{data_dir}{filename}', sr = None)

        # Zero crossing rates
        zrc = librosa.feature.zero_crossing_rate(song)[0]

        # Spectral centroid
        spc = librosa.feature.spectral_centroid(song, sr=sr)[0]

        # Spectral roll off
        spr = librosa.feature.spectral_rolloff(song, sr=sr)[0]

        # Calculate MFCCs
        mfcc = librosa.feature.mfcc(y=song, sr=sr)
        
        # Chroma frequencies
        chf = librosa.feature.chroma_stft(y=song, sr = sr)

        # print(f"{zrc[0].shape=}")
        # print(f"{spc.shape=}")
        # print(f"{spr.shape=}")
        # print(f"{mfcc.shape=}")
        # print(f"{chf.shape=}")
        
        to_append = f'{filename} {np.mean(zrc)} {np.mean(spc)} {np.mean(spr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        for e in chf:
            to_append += f' {np.mean(e)}'
        
        if include_genre:
            to_append += f' {genre}'
        
        writer.writerow(to_append.split())
        
        print(f"File complete {filename}")

train_dir = "./elec301-2021-music-genres/data/data/"
train_output_filename = 'train_data.csv'

gen(train_dir, train_output_filename, True)

test_dir = "./elec301-2021-music-genres/test_new/test_new/"
test_output_filename = 'test_data.csv'

gen(test_dir, test_output_filename, False)