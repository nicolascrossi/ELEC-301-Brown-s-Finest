import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import gen_mfcc_csv

#gen_mfcc_csv.create_librosa_mfccs()
reader = csv.reader(open("data.csv", "r"), delimiter=",")
x = list(reader)

# Remove header
x = x[1 :]
 
# Remove song name and convert genre to a number
genre_to_int = {}
int_to_genre = {}
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
val = 0
for g in genres:
    genre_to_int[g] = val
    int_to_genre[val] = g
    val += 1
for row in x:
    row.pop(0)
    row[-1] = genre_to_int[row[-1]]

data = np.array(x).astype("float")

X_train, X_test, y_train, y_test = train_test_split(data[:, 1 : -1], data[:, -1], test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# # Only the mfcc vals and their class
# x_data = data[ : , 1 :]

# x_train_data = x_data[np.random.choice(x_data.shape[0], 450, replace=False), :]
# x_train_vals = x_train_data[:, : -1]
# x_train_class = x_train_data[:, -1]

# print(x_train_vals.shape)
# print(x_train_class.shape)
# FOR NICO: comment out rest.
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
labels = neigh.predict(data)
labels = [int_to_genre[i] for i in labels]

print("training accuracies: ", train_acc)

header = 'filename label'
header = header.split()

file = open('submission.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for filename, label in zip(test_data, labels):
    file = open('submission.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow((filename[0], label))
#basic KNN

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
        #hidden_layer_sizes=(15,), random_state=1, max_iter=1000000)
#clf.fit(X_train, y_train)
