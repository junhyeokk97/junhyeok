from sklearn.datasets import load_diabetes
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

datasets = load_diabetes()
x= datasets.data
y= datasets.target
print(x.shape, y.shape)

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

# ======= KNeighborsRegressor =======
# acc:  0.345977065031307
# r2_score:  0.345977065031307