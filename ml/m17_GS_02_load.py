import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib 
from sklearn.metrics import mean_squared_error, r2_score


#1. 데이터 
x,y,= dataset = fetch_california_housing(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=55
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


n_split = 5
kfold = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = 55)


parameters =[
    {'n_estimators' : [100,500],'max_depth':[6,10,12],'learning_rate':[0.1,0.01,0.001]}, # 12
    {'max_depth':[6,8,10,12],'learning_rate':[0.1,0.01,0.001]},
    {'min_child_weight':[2,3,5,10],'learning_rate':[0.1,0.01,0.001]}
]

#2. 모델
path = './_save/m15_cv_results/'



model = joblib.load(path + 'm15_best_model2.joblib')


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
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))

print(model) # 모델 명세 출력 
print(type(model)) # 무슨 모델인지만 확인하고 싶을때 

# y_pred_best = model.best_estimator_.predict(x_test)


# 위나 아래나 별 차이 없음 쓰고 싶은거 쓰세요;;



# joblib.dump(model.best_estimator_,path + 'm15_best_model.joblib' )



# model.score :  0.8303525847499975
# Mean Squared Error:  0.22928213625073068
# R2 Score:  0.8303525847499975