import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from manage_csv import load_csv, write_submission

X, y, _ = load_csv("train_data.csv", True)

# Random state 0 = 57%
# Random state 100 = 58%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

log_reg = LogisticRegression(max_iter=1000000000)
log_reg.fit(X_train, y_train)
print(f'Train accuracy: {log_reg.score(X_train, y_train)}')
print(f'Test accuracy: {log_reg.score(X_test, y_test)}')

#TESTING

X_test, _, original = load_csv("test_data.csv", False)

labels = log_reg.predict(X_test)

write_submission("log_reg_submission.csv", labels, original)