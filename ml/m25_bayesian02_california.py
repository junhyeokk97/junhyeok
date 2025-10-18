import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import joblib
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,)
                                                    # stratify=y)

bayesian_params = {
    'learning_rate' : (0.001, 0.1),
    'max_depth' : (3, 10),
    'num_leaves' : (24, 40),
    'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 50),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 1),
    'max_bin' : (9, 500),
    'reg_lambda' : (0, 10),
    'rag_alpha' : (0.01, 50)
} 

def y_function(learning_rate, max_depth, num_leaves,
               min_child_samples, min_child_weight,
               subsample, colsample_bytree, max_bin,
               reg_lambda, rag_alpha):
    
    model = XGBRegressor(
    learning_rate=learning_rate,
    max_depth=int(max_depth),
    num_leaves=int(num_leaves),
    min_child_samples=int(min_child_samples),
    min_child_weight=min_child_weight,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    max_bin=int(max_bin),
    # reg_lambda=max(0, reg_lambda),
    reg_lambda=reg_lambda,
    rag_alpha=rag_alpha
)
    model.fit(x_train, y_train, early_stopping_rounds=15)
    y_pred = model.predict(x_test)
    return r2_score(y_test, y_pred)

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,     # 블랙박스 함수.
    pbounds=bayesian_params,
    random_state=50,
)

optimizer.maximize(init_points=5,    # init_point >> 최적화 시작 지점 또는 초기 시도 횟수
                   n_iter=30)

print(optimizer.max)


print('r2_score: ', optimizer.max['target'])