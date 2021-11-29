import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import gen_mfcc_csv

#gen_mfcc_csv.create_librosa_mfccs()
reader = csv.reader(open("data.csv", "r"), delimiter=",")
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
train_acc = []
for i in range(5, 15, 2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    dist, indices = neigh.kneighbors(X_test, return_distance=True)
    print("Mean distance: ", dist[0])

    #neigh.kneighbors(X, return_distance=False)

    
    train_acc.append(neigh.score(X_train, y_train))

    print(f"For {i} neighbors", "accuracy was:", neigh.score(X_train, y_train))
    print(f"For {i} neighbors", "test acc was:", neigh.score(X_test, y_test))

print("training accuracies: ", train_acc)


#basic KNN

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,), random_state=1)
clf.fit(X_train, y_train)