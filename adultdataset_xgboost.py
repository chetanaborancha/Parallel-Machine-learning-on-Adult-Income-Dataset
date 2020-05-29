import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start = time.time()

adult = pd.read_csv('adult_original_UCI_extended_final.csv')

for feature in adult.columns: # Loop through all columns in the dataframe
    if adult[feature].dtype == 'object': # Only apply for columns with categorical strings
        adult[feature] = pd.Categorical(adult[feature]).codes # Replace strings with an integer

X = adult.iloc[:, 0:-1]
y = adult.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,shuffle=1, random_state = 0)

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import time

xgdmat = xgb.DMatrix(X_train, y_train) # Create our DMatrix to make XGBoost more efficient

our_params = {'tree_method':'hist','eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':5, 'min_child_weight':3} 
# Grid Search CV optimized settings


cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 1000, nfold = 5,verbose_eval = 1,
                metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 300) # Look for early stopping that minimizes error
     
final_gb = xgb.train(our_params, xgdmat, num_boost_round = 930)           

print(cv_xgb.tail())


start_time = time.time()
our_params = {'tree_method':'hist','eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':5, 'min_child_weight':3} 
cpu_result = {}
final_gb = xgb.train(our_params, xgdmat, num_boost_round = 1300)
print("CPU Training Time: %s seconds" % (str(time.time() - start_time)))


for n_jobs in [2,4,8]:
    t1 = time.time()
    our_params1 = {'tree_method':'gpu_hist','eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':5, 'min_child_weight':3, 'n_jobs': n_jobs} 
    gpu_result = {} 
    final_gb1 = xgb.train(our_params1, xgdmat, num_boost_round = 1800)
    t2 = time.time()
    print ('with %d jobs: ' % n_jobs, t2 - t1)

testdmat = xgb.DMatrix(X_test)
from sklearn.metrics import accuracy_score
y_pred = final_gb.predict(testdmat) # Predict using our testdmat
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
print((accuracy_score(y_pred, y_test)*100), ((1-accuracy_score(y_pred, y_test))*100))


#import matplotlib.pyplot as plt
#xgb.plot_importance(final_gb)
importances = final_gb.get_fscore()
importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
print(importance_frame)
#importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')
#plt.savefig('adult dataset xgboost.png')
#plt.show(block=True)   










