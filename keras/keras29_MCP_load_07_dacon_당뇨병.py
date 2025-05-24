# dacon, 데이터 파일 별도
# https://dacon.io/competitions/official/236068/leaderboard

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

r = 55
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=r
)

from tensorflow.python.keras.models import load_model
path1 = './_save/keras28_mcp/07_dacon_당뇨병/'
model = load_model(path1 + 'k28_0604_1134_0167-0.4575.hdf5')


results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_predict)

# print(y_predict)
print('RanNo. :', r)
print('Loss   :', round(results[0], 4))
print('Acc    :', round(accuracy_score, 4))

# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
# submission_csv['Outcome'] = y_submit
# submission_csv.to_csv(path + 'submission_1.csv', index=False)


# #############
# RanNo. : 55
# Loss   : 0.5203
# Acc    : 0.7879

#############
# RanNo. : 55
# Loss   : 0.5203
# Acc    : 0.7879