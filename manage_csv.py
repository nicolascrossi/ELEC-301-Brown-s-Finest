import csv
from typing import Tuple
import numpy as np

def load_csv(filename: str, contains_labels: bool) -> Tuple[np.ndarray, np.ndarray, list]:
    '''
    Loads the given csv. If it contains data in the last column, pass True for contains_labels.

    Returns the feature data matrix, the labels, and the filenames.
    '''
    reader = csv.reader(open(filename, "r"), delimiter=",")
    orig = list(reader)

    # Remove header
    x = orig[1 :]
    filenames = []
    # Remove song name and convert genre to a number
    genre_to_int = {}
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    val = 0
    for g in genres:
        genre_to_int[g] = val
        val += 1
    for row in x:
        filenames.append(row.pop(0))
        if contains_labels:
            row[-1] = genre_to_int[row[-1]]

    data = np.array(x).astype("float")

    if contains_labels:
        X = data[:, 1 : -1]
        y = data[:, -1]
    else:
        X = data[:, 1 :]
        y = None
    
    return X, y, filenames

def get_int_to_genre() -> dict[int, str]:
    int_to_genre = {}
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    val = 0
    for g in genres:
            int_to_genre[val] = g
            val += 1

    return int_to_genre

def write_submission(filename: str, labels: np.ndarray, filenames: list):
    '''
    Using the given arguments creates a submission file.
    '''
    int_to_genre = get_int_to_genre()

    labels = [int_to_genre[i] for i in labels]

    header = 'filename label'
    header = header.split()

    file = open(filename, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)

    for name, label in zip(filenames, labels):
        file = open(filename, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow((name, label))