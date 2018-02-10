from __future__ import print_function
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print(iris_X[:2, :])
print(iris_y)


X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# predict V.S. real result
predict = knn.predict(X_test)
real_result = y_test

print(predict)
print(real_result)

result = real_result - predict
print(result)
print(1.0 - (np.count_nonzero(result) / np.size(result)))