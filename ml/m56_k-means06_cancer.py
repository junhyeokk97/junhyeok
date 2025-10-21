from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.cluster import KMeans
seed = 50
random.seed(seed)
np.random.seed(seed)

datasets = load_breast_cancer()
x= datasets.data
y= datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

scl = StandardScaler()
x_train = scl.fit_transform(x_train)
x_test = scl.transform(x_test)

model = KMeans(n_clusters=2, init='k-means++',
               n_init=10000, random_state=seed)
# ======= KMeans =======
# acc:  -1653.8249730226094 ??????????????????????????????????????????????
# accuracy_score:  0.10526315789473684 ???????????????????????????????????
# f1:  0.10526315789473684 ???????????????????????????????????????????????
y_train_pred = model.fit_predict(x_train)

print(y_train_pred[:10])    # [0 1 0 0 1 0 0 1 0 0]
print(y_train[:10])         # [1 0 1 1 0 0 1 0 1 1]

print("=======", model.__class__.__name__, "=======")
print('acc: ', model.score(x_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('accuracy_score: ', acc)

# r2 = r2_score(y_test, y_pred)
# print('r2_score: ', r2)

f1 = f1_score(y_test, y_pred, average='macro')
print('f1: ', f1)

# ======= KNeighborsRegressor =======
# acc:  0.9034920634920635
# r2_score:  0.9034920634920635