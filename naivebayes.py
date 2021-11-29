import scipy.io as sc
from scipy import signal, linalg
import numpy as np
import pandas as pd

#From Nico: for import
import csv

#K-Means methods
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB

#IMPORTING DATA FROM MASTER CSV -- Note: Already mirrors form of data matrix from HW7
reader = csv.reader(open("data.csv", "r"), delimiter=",")
X = list(reader)
#print(x[0])
#removes first header row
X = X[1 :]

# Remove song name, convert genre to a number
genre_to_int = {}
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
val = 0
for g in genres:
    genre_to_int[g] = val
    val += 1
for row in X:
    row.pop(0)
    row[-1] = genre_to_int[row[-1]]

#converting data and delineating into testing/training
data = np.array(X).astype("float") 
data = abs(data)
print(data.shape)


#x_train/test -- sets of 80/20% of mfcc data
#y_train/test -- sets of 80/20% of correct genres
x_train, x_test, y_train, y_test = train_test_split(data[:, 1 : -1], data[:, -1], test_size=0.2, random_state=0)

#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

nb = MultinomialNB()
nb.fit(x_train, y_train)

assigned_label = nb.predict(x_train)
predict_label = nb.predict(x_test)

train_acc = sum(assigned_label == y_train) / len(x_train)
test_acc = sum(predict_label == y_test) / len(x_test)

print("Training Accuracy: ", train_acc)
print("Testing Accuracy: ", test_acc)