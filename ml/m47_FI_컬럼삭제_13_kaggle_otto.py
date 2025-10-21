from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/kaggle/otto/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sampleSubmission.csv')

# 3. 데이터 분리
x = train.drop(['id', 'target'], axis=1)
y = train['target']
X_test = test.drop(['id'], axis=1)
test_ids = test['id']

# 4. 라벨 인코딩 및 스케일링
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 0 ~ 8 정수 인코딩
y = y_encoded.reshape(-1, 1)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_test_scaled = scaler.transform(x_test)

model = XGBClassifier(random_state=seed)
model.fit(x_train, y_train)
print("=======", model.__class__.__name__, "=======")
print('acc: ', model.score(x_test, y_test))
print(model.feature_importances_)

print(np.percentile(model.feature_importances_, 25))    # 0.0034034873824566603

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
print('acc: ', model.score(x_test, y_test)) # acc:  0.8152876535229476