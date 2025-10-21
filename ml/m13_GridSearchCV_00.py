import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,
                                                    stratify=y)

n_split = 5

kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

parameters = [
    {"C":[1,10,100,1000], "kernel":['linear', 'sigmoid'],
     'degree':[3,4,5]}, # 24
    {'C':[1,10,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},    # 6
    {'C':[1,10,100,1000], 'kernel':['sigmoid'],
     'gamma':[0.01,0.001,0.0001], 'degree':[3,4]}   # 24
]

#2. 모델
model = GridSearchCV(SVC(), parameters, cv=kfold,   # 54 * 5 = 270
                     verbose=1,
                     refit=True,    # test split 되지 않은 상태로 전체 테스트 1번 진행
                     n_jobs=-1)     # 총 train 271번

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
print('accuracy_score: ', accuracy_score(y_test, y_pred))
print('time : ', round(end-str))

# 최적의 매개변수:  SVC(C=1, kernel='linear')
# 최적의 파라미터:  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score:  0.9666666666666668
# model.score:  1.0
# accuracy_score:  1.0
# time :  3