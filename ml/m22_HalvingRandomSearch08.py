import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # 이것을 먼저 임포트
from sklearn.model_selection import HalvingRandomSearchCV
import time
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier,XGBRegressor
import pandas as pd 
from sklearn.metrics import r2_score

#1. 데이터 
path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 2. 불필요한 열 제거
train_csv = train_csv.drop(columns=['CustomerId', 'Surname'])
test_csv = test_csv.drop(columns=['CustomerId', 'Surname'])

# 3. 범주형 변수 라벨 인코딩
cat_cols = ['Geography', 'Gender']
for col in cat_cols:
    le = LabelEncoder()
    all_data = pd.concat([train_csv[col], test_csv[col]], axis=0)
    le.fit(all_data)
    train_csv[col] = le.transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

# 4. x, y 분리
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']



# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=55)


# (132027, 10)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



n_split = 5 
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)




parameters =[
    {'n_estimators' : [100,500],'max_depth':[6,10,12],'learning_rate':[0.1,0.01,0.001]}, # 12
    {'max_depth':[6,8,10,12],'learning_rate':[0.1,0.01,0.001]},
    {'min_child_weight':[2,3,5,10],'learning_rate':[0.1,0.01,0.001]}
]

#2. 모델
xgb = XGBClassifier()
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
                     min_resources = 20, # 1 iter때의 최소 훈련 행의 갯수
                     max_resources = 132027 # 데이터 행의 갯수(n_samples)

                     )

start = time.time()

#3. 훈련
model.fit(x_train,y_train)

end =time.time()

print('최적의 매개변수 : ',model.best_estimator_)
print('최적의 파라미터 : ',model.best_params_)

# exit() # Fitting 5 folds for each of 11 candidates, totalling 55 fits


#4 평가 에측

print('best_score : ',model.best_score_)
print('model.score : ',model.score(x_test,y_test))

y_pred= model.predict(x_test)
print('accuracy_score : ',accuracy_score(y_test,y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('best_acc_score :', accuracy_score(y_test, y_pred_best))

# 위나 아래나 별 차이 없음 쓰고 싶은거 쓰세요;;

print("걸린 시간:",round(end - start,2),'초')


path = './_save/m15_cv_results/'

print(pd.DataFrame(model.cv_results_).sort_values(
    'rank_test_score',
    ascending = True).to_csv(path + 'm22_00_rs_cv_08.csv',index = False)) 


import joblib

joblib.dump(model.best_estimator_,path + 'm22_08_best_model.joblib' )



# best_score :  0.8491346486673589
# model.score :  0.8630290544429969
# accuracy_score :  0.8630290544429969
# best_acc_score : 0.8630290544429969
# 걸린 시간: 7.5 초