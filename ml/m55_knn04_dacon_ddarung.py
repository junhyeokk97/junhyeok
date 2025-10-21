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
seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/dacon/따릉이/'      # .(점 한개) = 현재 작업폴더 study25

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # a=b b를 a에 넣겠다. // index_col : 이 컬럼은 인덱스다
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.dropna()       # train_csv에 결측치 데이터를 삭제 처리해라. 결측치 삭제하고 남은 데이터를 반환해서 덮어쓴다

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1) 
y = train_csv['count'] 

print(x.shape ,y.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

model = KNeighborsRegressor(n_neighbors=5)

model.fit(x_train, y_train)

print("=======", model.__class__.__name__, "=======")
print('acc: ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
# acc = accuracy_score(y_test, y_pred)
# print('accuracy_score: ', acc)

r2 = r2_score(y_test, y_pred)
print('r2_score: ', r2)

# f1 = f1_score(y_test, y_pred)
# print('f1: ', f1)

# acc:  0.7101339924133795
# r2_score:  0.7101339924133795
