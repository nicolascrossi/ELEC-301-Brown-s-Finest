import os
import csv

import librosa
import numpy as np


header = 'filename'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()


file = open('test_data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

#print(os.listdir('./elec301-2021-music-genres/data/data/'))
for filename in os.listdir('./elec301-2021-music-genres/test_new/test_new/'):

    song, sr = librosa.load(f'./elec301-2021-music-genres/test_new/test_new/{filename}')
    mfcc = librosa.feature.mfcc(y=song, sr=sr)
    to_append = f'{filename}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    file = open('test_data.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())