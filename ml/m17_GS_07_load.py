import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score

import joblib

#1. 데이터 
path = './_data/dacon/당뇨병/dacon diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submit_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

# 0값을 NaN으로 대체
train_csv[cols_to_fix] = train_csv[cols_to_fix].replace(0, np.nan)
test_csv[cols_to_fix] = test_csv[cols_to_fix].replace(0, np.nan)

# Feature Engineering (BMI * Age, Glucose / BMI)
train_csv['BMI*Age'] = train_csv['BMI'] * train_csv['Age']
train_csv['Glucose/BMI'] = train_csv['Glucose'] / (train_csv['BMI'] + 1e-5)
test_csv['BMI*Age'] = test_csv['BMI'] * test_csv['Age']
test_csv['Glucose/BMI'] = test_csv['Glucose'] / (test_csv['BMI'] + 1e-5)

# Drop target before preprocessing
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']


# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=55)


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
path = './_save/m15_cv_results/'



model = joblib.load(path + 'm15_best_model7.joblib')


#3. 훈련
# model.fit(x_train,y_train)


# print('최적의 매개변수 : ',model.best_estimator_)
# print('최적의 파라미터 : ',model.best_params_)

#4 평가 예측

# print('best_score : ',model.best_score_)

# Traceback (most recent call last):
#   File "c:\Study25\ml\m17_GS_00_load.py", line 54, in <module>
#     print('best_score : ',model.best_score_)
# AttributeError: 'XGBClassifier' object has no attribute 'best_score_'

# 그리드 서치로 저장했다고 생각하지만 xg부스트의 파라미터만(로) 저장 되어있음 



print('model.score : ',model.score(x_test,y_test))

y_pred= model.predict(x_test)
print('accuracy_score : ',accuracy_score(y_test,y_pred))




# 'c:\x5cStudy25\x5cml\x5cm17_GS_07_load.py' ;987ee169-26c2-4b8d-ad21-6580199b6cabmodel.score :  0.7709923664122137
# accuracy_score :  0.7709923664122137