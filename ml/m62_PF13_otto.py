import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

path = './_data/kaggle/santander/'

  # 실제 sample_submission.csv 위치
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# 2. 피처, 타겟 분리
x = train.drop(columns=['target', 'ID_code'])
y = train['target']
print(x.shape ,y.shape) 


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
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

# accuracy score: 0.16111111111111112
# f1 score: 0.07579322638146167
# time:  1.0998420715332031