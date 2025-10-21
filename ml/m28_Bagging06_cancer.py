import numpy as np
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor

seed =333
random.seed(seed)
np.random.seed(seed)

from xgboost import XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 

x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,stratify=y, random_state=333)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2 모델

# model = DecisionTreeRegressor()
model = BaggingRegressor(DecisionTreeRegressor(), 
                         n_estimators = 100,
                         n_jobs = -1,
                         random_state = 33,
                         bootstrap = False
                         )


# model = RandomForestRegressor(random_state=33)

#3 훈련
model.fit(x_train,y_train)

#4 평가 예측
results = model.score(x_test,y_test)
print('최종 점수: ', results)


# 최종 점수:  0.6771394841269842