import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

log_reg = LogisticRegression(max_iter=1000000)
log_reg.fit(X_train, y_train)
print(f'Train accuracy: {log_reg.score(X_train, y_train)}')
print(f'Test accuracy: {log_reg.score(X_test, y_test)}')