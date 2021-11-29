import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

reader = csv.reader(open("data.csv", "r"), delimiter=",")
x = list(reader)

# Remove header
x = x[1 :]

#FITTING

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

# Random state 0 = 57%
# Random state 100 = 58%
X_train, X_test, y_train, y_test = train_test_split(data[:, 1 : -1], data[:, -1], test_size=0.2, random_state=100)

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



#TESTING


reader = csv.reader(open("test_data.csv", "r"), delimiter=",")
test_data = list(reader)

# Remove header
test_data = test_data[1 :]

x = []
# Remove filename
for row in test_data:
    x.append(row[1:])

data = np.array(x).astype("float")

print(data.shape)

labels = log_reg.predict(data)

print(labels.shape)

labels = [int_to_genre[i] for i in labels]

print(labels[0:10])


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