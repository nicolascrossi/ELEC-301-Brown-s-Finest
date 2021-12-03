from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#from sklearn.experimental import enable_halving_search_cv
#from sklearn.model_selection import HalvingGridSearchCV
from scipy import stats
import numpy as np

from manage_csv import load_csv, write_submission

# Load the CSV. The third return value is the filenames list, but we don't care about it here
X, y, _ = load_csv("train_data10.csv", True)

#Simplifying X through PCA in order to avoid divergence when evaluating cross-validated model
pca = PCA(n_components=25)
X_small = pca.fit_transform(X)
print(X_small.shape)

X_train, X_test, y_train, y_test = train_test_split(X_small, y, test_size=0.2, random_state=100)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Here, both validated and not validated models are implemented in tandem, cv controls the number of folds (times the model is trained)
log_reg = LogisticRegressionCV(max_iter=100000000, cv=5, scoring='accuracy')
#log_reg = LogisticRegression(max_iter = 10000000000)
log_reg.fit(X_train, y_train)
print(f'Train accuracy: {log_reg.score(X_train, y_train)}')
print(f'Test accuracy: {log_reg.score(X_test, y_test)}')
