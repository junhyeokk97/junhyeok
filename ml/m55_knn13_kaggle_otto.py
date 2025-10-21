import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder


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

print(x.shape ,y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)

print("=======", model.__class__.__name__, "=======")
print('acc: ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc)

f1 = f1_score(y_test, y_pred, average = 'macro')
print('f1: ', f1)

# ======= KNeighborsClassifier =======
# acc:  0.7756948933419522
# accuracy_score:  0.7756948933419522
# f1:  0.7160044092274583