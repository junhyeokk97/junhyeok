from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder


seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/kaggle/santander/'

  # 실제 sample_submission.csv 위치
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# 2. 피처, 타겟 분리
x = train.drop(columns=['target', 'ID_code'])
y = train['target']
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

model = XGBClassifier(random_state=seed)
model.fit(x_train, y_train)
print("=======", model.__class__.__name__, "=======")
print('acc: ', model.score(x_test, y_test))
print(model.feature_importances_)

print(np.percentile(model.feature_importances_, 25))    # 0.002849584794603288

percentile = np.percentile(model.feature_importances_, 25)
# print(type(percentile)) <class 'numpy.float64'>

col_name=[]
for i, fi in enumerate(model.feature_importances_):
    # print(i, fi)
    if fi <= percentile:
        col_name.append(x.columns[i])
    else:
        continue
# print(col_name) 

x = pd.DataFrame(x, columns=x.columns)
x = x.drop(columns=col_name)

# print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

model.fit(x_train, y_train)
print('acc: ', model.score(x_test, y_test)) # acc:  0.91225