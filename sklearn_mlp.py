import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from manage_csv import load_csv, write_submission
from scipy import stats

# Load the CSV. The third return value is the filenames list, but we don't care about it here
X, y, _ = load_csv("train_data15.csv", True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

mlp = MLPClassifier()

mlp.fit(X_train, y_train)
print(f'Train accuracy: {mlp.score(X_train, y_train)}')
print(f'Test accuracy: {mlp.score(X_test, y_test)}')

#TESTING

# Load the CSV. The second return value is the labels, but it's none since our second argument is false
X_test, _, filenames = load_csv("test_data15.csv", False)

labels = mlp.predict(X_test)

# This is the number of chunks the original song was split into
split_count = 6
print(labels)
print(labels.shape)
labels = np.reshape(labels, (-1, split_count))
labels = stats.mode(np.transpose(labels))[0][0]
print(labels)
print(labels.shape)

reduced_filenames = []
for i in range(0, len(filenames), split_count):
    reduced_filenames.append(filenames[i][:-1])

write_submission("mlp_15.csv", labels, reduced_filenames)