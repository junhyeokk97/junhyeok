import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
x,y = load_digits(return_X_y=True)

n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

# model = HistGradientBoostingRegressor()
model = RandomForestRegressor()

scores = cross_val_score(model, x,y, cv=n_split)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))

# model = HistGradientBoostingRegressor()
# acc:  [0.81525952 0.80569468 0.83083017 0.86476136 0.77522311] 
# 평균 acc:  0.82

# model = RandomForestRegressor()
# acc:  [0.80462294 0.82284499 0.83474093 0.8531611  0.77780826] 
# 평균 acc:  0.82