from sklearn.datasets import fetch_california_housing

from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

model = XGBClassifier(random_state=seed)
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
print(col_name) # ['Pregnancies', 'BloodPressure']

x = pd.DataFrame(x, columns=x.columns)
x = x.drop(columns=col_name)

# print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

model.fit(x_train, y_train)
print('acc: ', model.score(x_test, y_test)) # acc:  0.696969696969697
