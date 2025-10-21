import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler

path = './_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(columns=['target'], axis=1)
y = train_csv['target']

n_split = 5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

# model = HistGradientBoostingRegressor()
model = RandomForestRegressor()

scores = cross_val_score(model, x,y, cv=n_split)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))

# model = HistGradientBoostingRegressor()
# acc:  [0.18587053 0.18691363 0.18993149 0.18973902 0.19402314] 
# 평균 acc:  0.19

# model = RandomForestRegressor()
