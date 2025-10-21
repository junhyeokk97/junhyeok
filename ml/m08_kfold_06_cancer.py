import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
import pandas as pd
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
x,y = load_breast_cancer(return_X_y=True)

n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

# model = HistGradientBoostingRegressor()
model = RandomForestRegressor()

scores = cross_val_score(model, x,y, cv=n_split)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))

# model = HistGradientBoostingRegressor()
# acc:  [0.75463378 0.82579093 0.92040732 0.89314211 0.82291245] 
# 평균 acc:  0.84

# model = RandomForestRegressor()
# acc:  [0.67838734 0.82857407 0.89510459 0.86361623 0.80755031] 
# 평균 acc:  0.81