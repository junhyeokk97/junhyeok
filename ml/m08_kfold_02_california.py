import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
x,y = fetch_california_housing(return_X_y=True)

n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
# kfold = StratifiedKFold(n_split=n_split, shuffle=True, random_state=50)       # StratifiedKFold는 분류에만 사용.

# model = HistGradientBoostingRegressor()
model = RandomForestRegressor()

scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함.
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))

# model = HistGradientBoostingRegressor()
# acc:  [0.82609537 0.83584096 0.8284281  0.84507414 0.84316244] 
# 평균 acc:  0.84

# model = RandomForestRegressor()
# acc:  [0.80422619 0.80552601 0.81274234 0.82429816 0.80837645] 
# 평균 acc:  0.81