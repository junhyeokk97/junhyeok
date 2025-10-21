from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import OneHotEncoder


seed = 50
random.seed(seed)
np.random.seed(seed)

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

y = y.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y)

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
        col_name.append(datasets.feature_names[i])
    else:
        continue
# print(col_name) 

x = pd.DataFrame(x, columns=datasets.feature_names)

x = x.drop(columns=col_name)

# print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

model.fit(x_train, y_train)
print('acc: ', model.score(x_test, y_test)) # acc:  0.8602762966553563
