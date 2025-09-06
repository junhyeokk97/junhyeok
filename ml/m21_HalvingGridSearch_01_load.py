import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import time
from xgboost import XGBClassifier,XGBRegressor
import pandas as pd 
from sklearn.metrics import r2_score,mean_squared_error
import joblib
import time


# 1. 데이터 로드
from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
n_split = 5



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
path = './_save/m15_cv_results/'



model = joblib.load(path + 'm20_01_best_model.joblib')


end =time.time()


# exit() # Fitting 5 folds for each of 11 candidates, totalling 55 fits


#4 평가 에측
print('model.score : ',model.score(x_test,y_test))

y_pred= model.predict(x_test)
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))

print(model) # 모델 명세 출력 
print(type(model))
