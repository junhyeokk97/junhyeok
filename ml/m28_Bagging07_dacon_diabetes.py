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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestRegressor

seed =333
random.seed(seed)
np.random.seed(seed)

from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
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

# (521, 10)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2 모델

# model = DecisionTreeRegressor()
model = BaggingClassifier(DecisionTreeClassifier(), 
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


# 최종 점수:  0.6870229007633588