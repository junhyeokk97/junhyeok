# cancer, dacon_당뇨병, kaggle_bank, wine, digits
# Decision, xgb, lgbm, cat,
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

path1 = './_data/dacon/diabetes/'
train_csv1 = pd.read_csv(path1 + 'train.csv', index_col=0)
test_csv1 = pd.read_csv(path1 + 'test.csv', index_col=0)

path2 = './_data/kaggle/bank/'
train_csv2 = pd.read_csv(path2+'train.csv', index_col=0)
test_csv2 = pd.read_csv(path2+'test.csv', index_col=0)

data_list = [load_breast_cancer, diabetes, bank, load_wine, load_digits]

scl_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]

model_list = [RandomForestClassifier(), XGBClassifier(), CatBoostClassifier(), LGBMClassifier()]

for i, data_set in enumerate(data_list):
    
    





x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, shuffle=True, random_state=333, stratify=y)

# scl = MinMaxScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# # 2. 모델
# model = RandomForestClassifier()

model = make_pipeline(StandardScaler(), RandomForestClassifier())
# model = make_pipeline(MinMaxScaler(), SVC())

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('score: ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

