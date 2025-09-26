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

# 1. 데이터 로드
# 1. 데이터 로드

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


parameters =[
    {'n_estimators' : [100,500],'max_depth':[6,10,12],'learning_rate':[0.1,0.01,0.001]}, # 12
    {'max_depth':[6,8,10,12],'learning_rate':[0.1,0.01,0.001]},
    {'min_child_weight':[2,3,5,10],'learning_rate':[0.1,0.01,0.001]}
]

#2. 모델
xgb = XGBRegressor()
# model = RandomizedSearchCV(xgb, parameters,cv = kfold, # 54 * 5 = 270번 
#                      verbose = 1,
#                      refit=True, # 1번
#                      n_jobs = -1, # 271번 
#                      n_iter=11, # 디폴트 10
#                      random_state=33

#                      )



model = HalvingRandomSearchCV(xgb, parameters, cv = kfold, # 54 * 5 = 270번 
                     verbose = 1,
                     refit=True, # 1번
                     n_jobs = -1, # 271번 
                     random_state=33,
                     factor = 3, # 배수: min_resources * factor //
                                 # n_candidates / factor
                     min_resources = 30, # 1 iter때의 최소 훈련 행의 갯수
                     max_resources = 16512 # 데이터 행의 갯수(n_samples)

                     )

start = time.time()

#3. 훈련
model.fit(x_train,y_train)

end =time.time()

print('최적의 매개변수 : ',model.best_estimator_)
print('최적의 파라미터 : ',model.best_params_)

# exit() # Fitting 5 folds for each of 11 candidates, totalling 55 fits


#4 평가 에측

#4 평가 에측

print('best_score : ',model.best_score_)
print('model.score : ',model.score(x_test,y_test))

print('best_score : ',model.best_score_)
print('model.score : ',model.score(x_test,y_test))

y_pred= model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2 score : ',r2)
print("걸린 시간:",round(end - start,2),'초')


path = './_save/m15_cv_results/'

print(pd.DataFrame(model.cv_results_).sort_values(
    'rank_test_score',
    ascending = True).to_csv(path + 'm22_00_rs_cv_02.csv',index = False)) 


import joblib

joblib.dump(model.best_estimator_,path + 'm22_02_best_model.joblib' )




# best_score :  0.7362223105813095
# model.score :  0.8187376188384602
# best_score :  0.7362223105813095
# model.score :  0.8187376188384602
# r2 score :  0.8187376188384602
# 걸린 시간: 6.17 초