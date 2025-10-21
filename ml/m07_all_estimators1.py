import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
x,y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test= train_test_split(x,y,
                                                   random_state=50,
                                                   test_size=0.2)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#2. 모델 구성
# model = RandomForestRegressor()
allAlgorithms = all_estimators(type_filter='regressor')
print('allAlgorithms: ', allAlgorithms)
print('모델: ', len(allAlgorithms))     # 55
print(type(allAlgorithms))

max_score = 0
max_name = 'name'
for (name, algorithm) in allAlgorithms:
    ######## 예외 처리 #######      try , except
    try:
        model = algorithm()
        
        #3. 훈련
        model.fit(x_train, y_train)
        #4. 평가, 예측
        results = model.score(x_test, y_test)
        print(name, 'score: ', results)
        if results > max_score:
            max_score = results
            max_name = name
    except:
        print(name, '은(는) 에러')
        
print("max model: ", max_name, max_score)

# max model:  HistGradientBoostingRegressor 0.8073475620072132