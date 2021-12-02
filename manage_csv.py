import csv
from typing import Tuple
import numpy as np

def load_csv(filename: str, contains_labels: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Loads the given csv. If it contains data in the last column, pass True for contains_labels.

    Returns the feature data matrix, the labels, and the original python array without the header in that order.
    '''
    reader = csv.reader(open(filename, "r"), delimiter=",")
    x = list(reader)

    # Remove header
    x = x[1 :]

    # Remove song name and convert genre to a number
    genre_to_int = {}
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    val = 0
    for g in genres:
        genre_to_int[g] = val
        val += 1
    for row in x:
        row.pop(0)
        if contains_labels:
            row[-1] = genre_to_int[row[-1]]

    data = np.array(x).astype("float")

    if contains_labels:
        X = data[:, 1 : -1]
        y = data[:, -1]
    else:
        X = data[:, 1 :]
        y = None
    
    return X, y, x

def get_int_to_genre() -> dict[int, str]:
    int_to_genre = {}
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    val = 0
    for g in genres:
            int_to_genre[val] = g
            val += 1

    return int_to_genre

def write_submission(filename: str, labels: np.ndarray, original: list):
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

    for row, label in zip(original, labels):
        file = open(filename, 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow((row[0], label))