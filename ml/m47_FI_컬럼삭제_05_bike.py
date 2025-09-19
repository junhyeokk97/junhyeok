from sklearn.datasets import fetch_california_housing

from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


seed = 50
random.seed(seed)
np.random.seed(seed)

path = 'c:/Study25/_data/kaggle/bike/'

# 데이터
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# -------------------------
# step 1+2. casual, registered 예측
# -------------------------
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv[['casual', 'registered']] #'registered 예측 추가'

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

model = XGBRegressor(random_state=seed)
model.fit(x_train, y_train)
print("=======", model.__class__.__name__, "=======")
print('acc: ', model.score(x_test, y_test))
print(model.feature_importances_)

print(np.percentile(model.feature_importances_, 25))

percentile = np.percentile(model.feature_importances_, 25)
# print(type(percentile)) <class 'numpy.float64'>

col_name=[]
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        col_name.append(x.columns[i])
    else:
        continue
print(col_name) # ['holiday', 'windspeed']

x = pd.DataFrame(x, columns=x.columns)
x = x.drop(columns=col_name)

# print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

model.fit(x_train, y_train)
print('r2: ', model.score(x_test, y_test)) # r2:  0.4378582239151001
