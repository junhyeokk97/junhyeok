import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
import pandas as pd
from sklearn.preprocessing import StandardScaler

path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()     # 클래스를 정의화 한다. > 인스턴스화 한다.
le_gen = LabelEncoder()
# train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
le_geo.fit(train_csv['Geography'])
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])

le_gen.fit(train_csv['Gender'])
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])

le_geo.fit(test_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(test_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)

y = train_csv['Exited']

n_split = 5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)

# model = HistGradientBoostingRegressor()
model = RandomForestRegressor()

scores = cross_val_score(model, x,y, cv=n_split)
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))


# model = HistGradientBoostingRegressor()
# acc:  [0.41728851 0.40485287 0.41612936 0.406341   0.40779391] 
# 평균 acc:  0.41

# model = RandomForestRegressor()
# acc:  [0.41742813 0.40575014 0.4155553  0.40617089 0.40729896]
# 평균 acc:  0.41