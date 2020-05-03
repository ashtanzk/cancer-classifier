import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data # all features
y = cancer.target # all labels/targets

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

# adding a kernel using SVM
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print("SVM accuracy: ", acc)


# testing using KNN to compare
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
acc_knn = metrics.accuracy_score(y_test, y_pred)
print("KNN accuracy: ", acc_knn)