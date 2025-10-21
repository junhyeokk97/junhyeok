import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
x,y = load_wine(return_X_y=True)

n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

# model = HistGradientBoostingRegressor()
model = RandomForestRegressor()

scores = cross_val_score(model, x,y, cv=n_split)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))


# model = HistGradientBoostingRegressor()
# acc:  [0.         0.54317683 0.         0.78872365 0.        ]
# 평균 acc:  0.27

# model = RandomForestRegressor()
# acc:  [0.         0.73534582 0.         0.82843881 0.        ] 
# 평균 acc:  0.31