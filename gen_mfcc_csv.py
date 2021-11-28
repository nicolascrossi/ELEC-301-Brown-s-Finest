import os
import csv

import librosa
import numpy as np

def create_librosa_mfccs():
    header = 'song number'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' genre label'
    header = header.split()


    file = open('data.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

    #print(os.listdir('../elec301-2021-music-genres/data/data/'))
    for filename in os.listdir('../elec301-2021-music-genres/data/data/'):
        num_idx = 0
        while not filename[num_idx].isdigit():
            num_idx += 1
        end_idx = num_idx
        while filename[end_idx].isdigit():
            end_idx += 1
        genre = filename[: num_idx]
        song_num = filename[num_idx : end_idx + 1]

        song, sr = librosa.load(f'../elec301-2021-music-genres/data/data/{filename}')
        mfcc = librosa.feature.mfcc(y=song, sr=sr)
        to_append = f'{filename} {song_num}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {genre}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

if '__name__' == "__main__":
    create_librosa_mfccs()
