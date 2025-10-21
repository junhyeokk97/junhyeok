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
import warnings
import joblib
warnings.filterwarnings('ignore')
#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,)
                                                    # stratify=y)

n_split = 5

# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)
kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12],
     'learning_rate':[0.1 , 0.01, 0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1 , 0.01, 0.001]}, # 12
    {'min_child_weight':[2,3,4,5,10], 'learning_rate':[0.1 , 0.01, 0.001]}  # 15
]

# #2. 모델
# # xgb = XGBClassifier()
# xgb = XGBRegressor()
# model = GridSearchCV(xgb, parameters, cv=kfold,
#                      verbose=1,
#                      refit=True,    # test split 되지 않은 상태로 전체 테스트 1번 진행
#                      n_jobs=-1)     # 총 train 271번

path = './_save/m15_cv_results/'
model = joblib.load(path + 'm15_best_model_02.joblib')

#3. 훈련
# str = time.time()
# model.fit(x_train, y_train)
# end = time.time()

# print('최적의 매개변수: ', model.best_estimator_)
# print('최적의 파라미터: ', model.best_params_)

#4. 평가, 예측
# print('best_score: ', model.best_score_)    # train에서의 최고 성능.
print('model.score: ', model.score(x_test,y_test))
  
y_pred = model.predict(x_test)
print('r2_score: ', r2_score(y_test, y_pred))
# print('time : ', round(end-str))

# path = './_save/m15_cv_results/'
# joblib.dump(model.best_estimator_, path + 'm15_best_model_02.joblib')

# model.score:  0.8402408512675119
# r2_score:  0.8402408512675119