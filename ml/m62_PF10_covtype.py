import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
import time

plt.rcParams['font.family'] = 'Malgun Gothic'

seed = 50
random.seed(seed)
np.random.seed(seed)

dataset = fetch_covtype()
x = dataset.data
y = dataset.target


print(x.shape ,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf_tr = pf.fit_transform(x_train)
x_pf_ts = pf.transform(x_test)

scl = StandardScaler()
x_pf_tr = scl.fit_transform(x_pf_tr)
x_pf_ts = scl.transform(x_pf_ts)

print(x_pf_tr.shape ,y.shape)

model = LinearRegression()

str = time.time()
model.fit(x_pf_tr, y_train)
end = time.time()
print("=======", model.__class__.__name__, "=======")
# print('acc: ', model.score(x_pf_ts, y_test))

y_pred = model.predict(x_pf_ts)
y_pred_binary = (y_pred > 0.5).astype(int)
# acc = accuracy_score(y_test, y_pred)
# print('accuracy_score: ', acc)

acc = accuracy_score(y_test, y_pred_binary)
#f1 다중에서도 사용 가능!!!
f1 = f1_score(y_test, y_pred_binary, average = 'macro')
print('accuracy score:', acc)
print('f1 score:', f1)
print('time: ', end-str)

# accuracy score: 0.3622594747168772
# f1 score: 0.06649648688267705
# time:  70.3228530883789