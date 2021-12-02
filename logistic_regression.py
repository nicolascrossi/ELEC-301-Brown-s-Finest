from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

from manage_csv import load_csv, write_submission

# Load the CSV. The third return value is the filenames list, but we don't care about it here
X, y, _ = load_csv("train_data.csv", True)

# Random state 0 = 57%
# Random state 100 = 58%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

log_reg = LogisticRegression(max_iter=1000000000)
log_reg.fit(X_train, y_train)
print(f'Train accuracy: {log_reg.score(X_train, y_train)}')
print(f'Test accuracy: {log_reg.score(X_test, y_test)}')

#TESTING

# Load the CSV. The second return value is the labels, but it's none since our second argument is false
X_test, _, filenames = load_csv("test_data.csv", False)

labels = log_reg.predict(X_test)

# This is the number of chunks the original song was split into
split_count = 15
print(labels)
print(labels.shape)
labels = np.reshape(labels, (-1, split_count))
labels = stats.mode(np.transpose(labels))[0][0]
print(labels)
print(labels.shape)

reduced_filenames = []
for i in range(0, len(filenames), split_count):
    reduced_filenames.append(filenames[i][:-1])

write_submission("log_reg_submission.csv", labels, reduced_filenames)