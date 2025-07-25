# train.csv와 new_test.csv로 count 예측

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

path = './_data/kaggle/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'new_test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'samplesubmission.csv')

print(train_csv)
print(train_csv.shape)
print(test_csv.shape)
print(submission_csv.shape)

print(train_csv.columns)
print(test_csv.columns)
print(submission_csv.columns)

print(train_csv.isnull().sum())
print('#######')
print(test_csv.isna().sum())
print('#######')
print(train_csv.describe())

x = train_csv.drop(['count'], axis= 1)
y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=55)


     
model = Sequential()
model.add(Dense(512, activation='relu', input_dim= 10))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))       


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=3)

loss = model.evaluate(x_test, y_test)

results = model.predict(x_test)
r2 = r2_score(y_test, results)

def RMSE(y_test,results):
    return np.sqrt(mean_squared_error(y_test, results))
rmse = RMSE(y_test, results)

print('loss: ', loss)
print('r2 : ', r2)
print('RMSE: ', rmse)

y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit

# submission_csv.to_csv( path + 'submission_0522_1635.csv', index=False) 