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
from catboost import CatBoostRegressor
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


def cat_tuning(learning_rate, depth, l2_leaf_reg, subsample):
    params = {
        'learning_rate': learning_rate,
        'depth': int(depth),                        # CatBoost는 depth를 int로 받음
        'l2_leaf_reg': l2_leaf_reg,
        'subsample': subsample,
        'iterations': 100,
        'random_seed': 333,
        'verbose': 0
    }

    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    score = r2_score(y_test, pred)
    return score

# 3. 탐색할 하이퍼파라미터 범위
bayesian_params = {
    'learning_rate': (0.01, 0.3),
    'depth': (4, 10),                    # int로 변환 필요
    'l2_leaf_reg': (1, 10),
    'subsample': (0.5, 1.0)
}




# def y_function(x1,x2):
#     return -x1 **2 - (x2 -2) **2 + 10

# pip install bayesian-optimization



optimizer = BayesianOptimization(
    f = cat_tuning,
    pbounds=bayesian_params,
    random_state=333,
    verbose=2
)



optimizer.maximize(init_points = 15,
                   n_iter = 30)

print("Best parameters:")
print(optimizer.max)



