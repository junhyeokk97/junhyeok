import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

plt.rcParams['font.family'] = 'Malgun Gothic'

seed = 50
random.seed(seed)
np.random.seed(seed)

path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']


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

f1 = f1_score(y_test, y_pred_binary, average = 'macro')
print('f1 score:', f1)
print('time: ', end-str)

# f1 score: 0.7555555555555555
# time:  0.000997781753540039