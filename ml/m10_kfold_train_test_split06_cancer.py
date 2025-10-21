import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pandas as pd

x,y = load_breast_cancer(return_X_y=True)

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
# model = MLPRegressor(max_iter=1000)

scores = cross_val_score(model, x_train, y_train, cv=n_split)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))

# model = HistGradientBoostingRegressor()
# acc:  [0.75463378 0.82579093 0.92040732 0.89314211 0.82291245] 
# 평균 acc:  0.84

# model = RandomForestRegressor()
# acc:  [0.67838734 0.82857407 0.89510459 0.86361623 0.80755031] 
# 평균 acc:  0.81

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC: ', acc)

# model = MLPClassifier(max_iter=1000)
# acc:  [0.96703297 0.92307692 0.93406593 0.94505495 0.94505495]
# 평균 acc:  0.94
# cross_val_predict ACC:  0.7368421052631579

