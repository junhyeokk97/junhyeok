import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor

x,y = load_boston(return_X_y=True)

n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

model = HistGradientBoostingRegressor()
# model = RandomForestRegressor()

scores = cross_val_score(model, x,y, ev=kfold)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))