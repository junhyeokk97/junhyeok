import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
x,y = fetch_covtype(return_X_y=True)

n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

# model = HistGradientBoostingRegressor()
model = RandomForestRegressor()

scores = cross_val_score(model, x,y, cv=n_split)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))


# model = HistGradientBoostingRegressor()
# acc:  [0.58702647 0.03515907 0.3473995  0.38368977 0.25265914] 
# 평균 acc:  0.32

# model = RandomForestRegressor()
# acc:  [0.71370828 0.24892956 0.50087728 0.5180989  0.4985381 ] 
# 평균 acc:  0.5