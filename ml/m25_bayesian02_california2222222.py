import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier, XGBRegressor
import random
seed = 50
random.seed(seed)
np.random.seed(seed)
from xgboost.callback import EarlyStopping
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

bayesian_params = {
    'n_estimators' : (100, 500),
    'learning_rate' : (0.0001, 0.5),
    'max_depth' : (3, 10),
    # 'num_leaves' : (24, 40),
    # 'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 10),
    'gamma' : (0, 5),
    'subsample' : (0.5, 2),     # 0~1 사이에.
    'colsample_bytree' : (0.5, 1),
    'colsample_bylevel' : (0.5, 1),
    # 'max_bin' : (9, 500),
    'reg_lambda' : (0, 100),
    'reg_alpha' : (0, 10)
} 

def xgb_hamsu(n_estimators, learning_rate, max_depth, min_child_weight,gamma 
              ,subsample, colsample_bytree, colsample_bylevel, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : int(n_estimators),
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'min_child_weight' : int(round(min_child_weight)),
        'gamma' : gamma,
        'subsample' : max(min(subsample,1),0),       # 0~1 사이로 형성.
        'colsample_bytree' : colsample_bytree,
        'colsample_bylevel' : colsample_bylevel,
        'reg_lambda' : max(reg_lambda, 0),      # dafault : 1 // L2 정규화 //릿지
        'reg_alpha' : reg_alpha                 # default : 0 // L1 정규화 // 라쏘
    }
    model = XGBRegressor(**params, n_jobs=-1)
    
    
    
    model.fit(x_train, y_train,
              eval_set = [(x_test, y_test)],
              verbose=0,
              eval_metric='rmse',
              early_stopping_rounds=15)
    y_pred = model.predict(x_test)
    results = r2_score(y_test, y_pred)
    return results

optimizer = BayesianOptimization(
    f = xgb_hamsu,     # 블랙박스 함수.
    pbounds=bayesian_params,
    random_state=seed,
)

n_iter = 300
str = time.time()
optimizer.maximize(init_points=5,    # init_point >> 최적화 시작 지점 또는 초기 시도 횟수
                   n_iter=n_iter)
end = time.time()
print(optimizer.max)
print('걸린 시간: ', round(end-str,2))

# {'target': 0.8356200097413916, 'params': {'n_estimators': 292.98988752436605, 'learning_rate': 0.5, 'max_depth': 5.959774619970895, 'min_child_weight': 2.307814042055676, 'gamma': 0.0, 'subsample': 2.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 0.5, 'reg_lambda': 71.85835589522557, 'reg_alpha': 3.876451804020859}}
# 걸린 시간:  1014.39