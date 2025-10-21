import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

datasets = fetch_california_housing()

x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,
                                                    shuffle=True,)

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_test = std.transform(x_test)

n_split = 5

kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
# kfold = StratifiedKFold(n_split=n_split, shuffle=True, random_state=50)       # StratifiedKFold는 분류에만 사용.

# model = HistGradientBoostingRegressor()
# model = RandomForestRegressor()
# model = MLPClassifier()
model = MLPRegressor()

scores = cross_val_score(model, x_train, y_train, cv=kfold) # fit까지 포함.
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_pred)
print('cross_val_predict ACC: ', r2)

# model = HistGradientBoostingRegressor()
# acc:  [0.82609537 0.83584096 0.8284281  0.84507414 0.84316244] 
# 평균 acc:  0.84

# model = RandomForestRegressor()
# acc:  [0.80422619 0.80552601 0.81274234 0.82429816 0.80837645] 
# 평균 acc:  0.81

# model = MLPRegressor()
# acc:  [0.75314713 0.77646117 0.76956041 0.78441966 0.77434542]
# 평균 acc:  0.77
# cross_val_predict ACC:  0.43226377946892935