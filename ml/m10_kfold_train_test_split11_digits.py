import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

x,y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,
                                                    shuffle=True)

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_test = std.transform(x_test)

n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

# model = HistGradientBoostingRegressor()
# model = RandomForestRegressor()
model = MLPClassifier(max_iter=1000)
# model = HistGradientBoostingClassifier()
# model = MLPRegressor(max_iter=1000)

scores = cross_val_score(model, x_train, y_train, cv=n_split)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))

# model = HistGradientBoostingRegressor()
# acc:  [0.81525952 0.80569468 0.83083017 0.86476136 0.77522311] 
# 평균 acc:  0.82

# model = RandomForestRegressor()
# acc:  [0.80462294 0.82284499 0.83474093 0.8531611  0.77780826] 
# 평균 acc:  0.82

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC: ', acc)

# model = MLPClassifier(max_iter=1000)
# acc:  [0.97916667 0.97569444 0.9825784  0.9825784  0.9825784]
# 평균 acc:  0.98
# cross_val_predict ACC:  0.925