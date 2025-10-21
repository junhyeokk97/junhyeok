import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)

model = MLPRegressor()


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
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        #4. 평가, 예측
        y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
        r2 = r2_score(y_test, y_pred)
        print('################################')
        print(f'{name}의 cross_val_score   정답률:')
        print(np.round(np.mean(scores),4))
        print(f'{name}의 cross_val_predict 정답률:')
        print(np.round(r2,4))
    except:
        print(name, '은(는) 에러')
        
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))
print('cross_val_predict ACC: ', r2)

# acc:  [0.32031665 0.35258254 0.34674644 0.34668211 0.34682057]
# 평균 acc:  0.34
# cross_val_predict ACC:  -0.12481918976578932