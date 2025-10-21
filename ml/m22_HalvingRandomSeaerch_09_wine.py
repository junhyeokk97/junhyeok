import numpy as np
from sklearn.datasets import load_wine
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
import warnings
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
import joblib

#1. 데이터
x,y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,)
                                                    # stratify=y)

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_test = std.transform(x_test)

n_split = 5

# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)
kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12],
     'learning_rate':[0.1 , 0.01, 0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1 , 0.01, 0.001]}, # 12
    {'min_child_weight':[2,3,4,5,10], 'learning_rate':[0.1 , 0.01, 0.001]}  # 15
]

#2. 모델
# xgb = XGBClassifier()
xgb = XGBRegressor()   # 총 train 271번

model = HalvingRandomSearchCV(xgb, parameters, cv=kfold,   # 54 * 5 = 270
                     verbose=1,
                     refit=True,    # test split 되지 않은 상태로 전체 테스트 1번 진행
                     n_jobs=-1,
                     random_state=50,
                    #  n_iter=11,
                      factor = 2,    # 데이터는 세 배로 늘리고, 후보군 3분의 1로.
                                    # 훈련 횟수와 상관 없이 모든 데이터는 중복 된다.
                                    # 배수 : min_resource * factor
                                    #       n_candidates / factor
                     min_resources=30,  # 1 iter 때의 최소 훈련 행의 수
                     max_resources=180, # 데이터 행의 개수(n_sample)
                     )

#3. 훈련
str = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수: ', model.best_estimator_)
print('최적의 파라미터: ', model.best_params_)

#4. 평가, 예측
print('best_score: ', model.best_score_)    # train에서의 최고 성능.
print('model.score: ', model.score(x_test,y_test))

y_pred = model.predict(x_test)
print('r2_score: ', r2_score(y_test, y_pred))
print('time : ', round(end-str))

path = './_save/m15_cv_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score',    # rank_test_score 기준으로 오름차순 정렬.
                                                  ascending=True).to_csv(path + 'm22_09_rs_cv_results.csv')

path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_, path + 'm22_09_best_model.joblib')  # 가중치 저장.


# 최적의 매개변수:  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,  
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.1, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None, 
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=5, missing=nan, monotone_constraints=None,
#              multi_strategy=None, n_estimators=None, n_jobs=None,
#              num_parallel_tree=None, random_state=None, ...) 
# 최적의 파라미터:  {'learning_rate': 0.1, 'min_child_weight': 
# 5}
# best_score:  0.9039352655410766
# model.score:  0.9593285322189331
# r2_score:  0.9593285322189331
# time :  6



# n_iterations: 3
# n_required_iterations: 6
# n_possible_iterations: 3
# min_resources_: 30
# max_resources_: 180
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 45
# n_resources: 30
# Fitting 5 folds for each of 45 candidates, totalling 225 fits
# ----------
# iter: 1
# n_candidates: 23
# n_resources: 60
# Fitting 5 folds for each of 23 candidates, totalling 115 fits
# ----------
# iter: 2
# n_candidates: 12
# n_resources: 120
# Fitting 5 folds for each of 12 candidates, totalling 60 fits
# 최적의 매개변수:  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.1, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=10, missing=nan, monotone_constraints=None,
#              multi_strategy=None, n_estimators=None, n_jobs=None,
#              num_parallel_tree=None, random_state=None, ...)
# 최적의 파라미터:  {'learning_rate': 0.1, 'min_child_weight': 10}
# best_score:  0.9005213355592974
# model.score:  0.9512666283055198
# r2_score:  0.9512666283055198
# time :  8





# n_iterations: 3
# n_required_iterations: 3
# n_possible_iterations: 3
# min_resources_: 30
# max_resources_: 180
# aggressive_elimination: False
# factor: 2
# ----------
# iter: 0
# n_candidates: 6
# n_resources: 30
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# ----------
# iter: 1
# n_candidates: 3
# n_resources: 60
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# ----------
# iter: 2
# n_candidates: 2
# n_resources: 120
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# 최적의 매개변수:  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.01, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=2, missing=nan, monotone_constraints=None,
#              multi_strategy=None, n_estimators=None, n_jobs=None,
#              num_parallel_tree=None, random_state=None, ...)
# 최적의 파라미터:  {'min_child_weight': 2, 'learning_rate': 0.01}
# best_score:  0.7576519625753171
# model.score:  0.8287430219976514
# r2_score:  0.8287430219976514
# time :  6