import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

path = './_data/kaggle/bike'
train_csv = pd.read_csv(path + '/train.csv', index_col=0)
test_csv =  pd.read_csv(path + '/test.csv', index_col=0)
submission_csv = pd.read_csv(path + '/sampleSubmission.csv')

x = train_csv.drop(columns=['casual','registered','count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,
                                                    shuffle=True)

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_test = std.transform(x_test)

n_split=5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

# model = HistGradientBoostingRegressor()
# model = RandomForestRegressor()
# model = MLPClassifier(max_iter=1000)
model = MLPRegressor(max_iter=1000)

scores = cross_val_score(model, x_train, y_train, cv=n_split)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))

# model = HistGradientBoostingRegressor()
# acc:  [-0.26492924 -0.12048128 -0.12826948  0.12810115  0.10672117] 
# 평균 acc:  -0.06

# model = RandomForestRegressor()
# acc:  [-0.71754739 -0.51635905 -0.37028264  0.07962749  0.07874332] 
# 평균 acc:  -0.29

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_pred)
print('cross_val_predict ACC: ', r2)

# model = MLPRegressor(max_iter=1000)
# acc:  [0.2893252  0.28881509 0.29484354 0.30129169 0.29469956]
# 평균 acc:  0.29
# cross_val_predict ACC:  0.2677423111928826