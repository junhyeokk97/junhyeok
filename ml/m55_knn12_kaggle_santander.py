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

path = './_data/kaggle/santander/'

  # 실제 sample_submission.csv 위치
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# 2. 피처, 타겟 분리
x = train.drop(columns=['target', 'ID_code'])
y = train['target']
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
# acc:  0.8994
# accuracy_score:  0.8994
# f1:  0.47549257295397784