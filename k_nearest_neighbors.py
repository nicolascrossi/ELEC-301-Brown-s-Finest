import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import normalize

from manage_csv import load_csv, write_submission
from scipy import stats

# #gen_mfcc_csv.create_librosa_mfccs()
# reader = csv.reader(open("data.csv", "r"), delimiter=",")
# x = list(reader)

# # Remove header
# x = x[1 :]
 
# # Remove song name and convert genre to a number
# genre_to_int = {}
# int_to_genre = {}
# genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
# val = 0
# for g in genres:
#     genre_to_int[g] = val
#     int_to_genre[val] = g
#     val += 1
# for row in x:
#     row.pop(0)
#     row[-1] = genre_to_int[row[-1]]

# data = np.array(x).astype("float")

# Load the CSV. The third return value is the filenames list, but we don't care about it here
X, y, _ = load_csv("train_data1.csv", True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = normalize(X_train)
X_test = normalize(X_test)

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
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)

print(f'Train accuracy: {neigh.score(X_train, y_train)}')
print(f'Test accuracy: {neigh.score(X_test, y_test)}')

#TESTING


# reader = csv.reader(open("test_data.csv", "r"), delimiter=",")
# test_data = list(reader)

# # Remove header
# test_data = test_data[1 :]

# x = []
# # Remove filename
# for row in test_data:
#     x.append(row[1:])

# data = np.array(x).astype("float")

# Load the CSV. The second return value is the labels, but it's none since our second argument is false
data, _, filenames = load_csv("test_data1.csv", False)

data = normalize(data)

print(data.shape)

labels = neigh.predict(data)

# print(labels.shape)

# labels = [int_to_genre[i] for i in labels]

# print(labels[0:10])

# This is the number of chunks the original song was split into
split_count = 1
print(labels)
print(labels.shape)
labels = np.reshape(labels, (-1, split_count))
labels = stats.mode(np.transpose(labels))[0][0]
print(labels)
print(labels.shape)

reduced_filenames = []
for i in range(0, len(filenames), split_count):
    reduced_filenames.append(filenames[i][:-1])

write_submission("knn_normed_submission_1_5n.csv", labels, reduced_filenames)


# header = 'filename label'
# header = header.split()

# file = open('submission.csv', 'w', newline='')
# with file:
#     writer = csv.writer(file)
#     writer.writerow(header)

# for filename, label in zip(test_data, labels):
#     file = open('knn_normed.csv', 'a', newline='')
#     with file:
#         writer = csv.writer(file)
#         writer.writerow((filename[0], label))
# #basic KNN

# #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#         #hidden_layer_sizes=(15,), random_state=1, max_iter=1000000)
# #clf.fit(X_train, y_train)
