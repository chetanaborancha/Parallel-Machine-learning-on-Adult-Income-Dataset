from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
 # Feature Importance
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
import os, time, random


adult = pd.read_csv('adult_original_UCI_extended_final.csv')

adult.replace([' Divorced', ' Married-AF-spouse', 
              ' Married-civ-spouse', ' Married-spouse-absent', 
              ' Never-married',' Separated',' Widowed'],
             [' divorced',' married',' married',' married',
              ' not married',' not married',' not married'], inplace = True)
adult.head(10)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
category_col =['employer', 'race','marital', 'sex', 'income'] 

for col in category_col:
    adult[col] = labelEncoder.fit_transform(adult[col].astype(str))
    
category_col_1 =['employer', 'edu', 'occupation',
               'relationship','country'] 

adult = pd.get_dummies(adult, columns=category_col_1, drop_first=True)

adult=adult.drop('fnlwt',1)
adult =adult[[c for c in adult if c not in ['income']] + ['income']]

X = adult.iloc[:, 0:-1]
y = adult.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=69)

from sklearn.ensemble import RandomForestClassifier
# n_estimators is the amount of trees to build
classifier=RandomForestClassifier(n_estimators=25)


# fit the RandomForest Model
classifier=classifier.fit(X_train,y_train)
# prediction scoring of the model (array of binary 0-1)
predictions=classifier.predict(X_test)


# confusion matrix / missclassification matrix
print(sklearn.metrics.confusion_matrix(y_test,predictions))
print(sklearn.metrics.accuracy_score(y_test, predictions))

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X_train,y_train)
# display the relative importance of each attribute
print(model.feature_importances_)

print(max(model.feature_importances_))
max_val = np.where(model.feature_importances_ == max(model.feature_importances_))

min_val = np.where(model.feature_importances_ == min(model.feature_importances_))

print(max_val, min_val)

import time
clf = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=2)
t1 = time.time()
ytest = clf.fit(X_train, y_train).predict(X_test)
t2 = time.time()
print ('no jobs: ', t2 - t1)
for n_jobs in [2,4,8]:
    t1 = time.time()
    clf = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=2, n_jobs=n_jobs)
    ytest = clf.fit(X_train, y_train).predict(X_test)
    t2 = time.time()
    print ('with %d jobs: ' % n_jobs, t2 - t1)

