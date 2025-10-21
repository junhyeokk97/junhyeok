from sklearn.datasets import fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import OneHotEncoder

seed = 50
random.seed(seed)
np.random.seed(seed)

datasets = fetch_covtype()
x= datasets.data
y= datasets.target

y = y.reshape(-1, 1)
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

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
# acc:  0.92890089842002
# accuracy_score:  0.92890089842002
# f1:  0.8781430970670414