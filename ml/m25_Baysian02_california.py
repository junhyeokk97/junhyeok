from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # 이것을 먼저 임포트
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

import time
from xgboost import XGBClassifier,XGBRegressor
import pandas as pd 
from sklearn.metrics import r2_score


x, y = fetch_california_housing(return_X_y=True)

# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=55)


# (16512, 8)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


n_split = 5
# kfold = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = 55)
kfold = KFold(n_splits=5, shuffle=True, random_state=333)



def xgb_tuning(learning_rate, max_depth, num_leaves,
                 min_child_samples, min_child_weight,
                 subsmaple, colsample_bytree, max_bin,
                 reg_lambda, reg_alpha):
    
    params = {
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        # 'num_leaves': int(num_leaves),
        # 'min_child_samples': int(min_child_samples),
        'min_child_weight': min_child_weight,
        'subsample': subsmaple,
        'colsample_bytree': colsample_bytree,
        'max_bin': int(max_bin),
        'reg_lambda': max(reg_lambda, 0),  # 음수 방지
        'reg_alpha': reg_alpha,
        'n_estimators': 100,
        'random_state': 333,
    }
    scores=[]
    model = XGBRegressor(**params)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = r2_score(y_test, pred)
    scores.append(score)
        
    return np.mean(scores)
 
 
 
    
bayesian_params = {
               'learning_rate' : (0.001,0.1),
               'max_depth' : (3,10),
               'num_leaves' : (24,40),
               'min_child_samples' : (10,200),
               'min_child_weight' : (1,50),
               'subsmaple' : (0.5,1),
               'colsample_bytree' : (0.5,1),
               'max_bin' : (9,500),
               'reg_lambda' : (-0.001,10),
               'reg_alpha' : (0.01,50)
               
               }




# def y_function(x1,x2):
#     return -x1 **2 - (x2 -2) **2 + 10

# pip install bayesian-optimization



optimizer = BayesianOptimization(
    f = xgb_tuning,
    pbounds=bayesian_params,
    random_state=333,
    verbose=2
)



optimizer.maximize(init_points = 15,
                   n_iter = 30)

print("Best parameters:")
print(optimizer.max)



