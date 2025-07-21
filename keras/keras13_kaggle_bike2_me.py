from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

path = './_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(train_csv)
# print(test_csv)
# exit()

x = train_csv.drop(['casual','registered', 'count'], axis=1)
y = train_csv[['casual','registered',]]

# print(x)    #[10886 rows x 8 columns]
# print(y)    #[10886 rows x 2 columns]

import random as rd
n = rd.randint(1, 100000)
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=n)

model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(230, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(130, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=32)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
def RMSE(a, b) : return np.sqrt(mean_squared_error(a, b))
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

y_new_test = model.predict(test_csv)
test_csv[['casual','registered']] = y_new_test
# print(test_csv)     #[6493 rows x 10 columns]
file = 'new_test_1.csv'
test_csv.to_csv(path + file, index=False)


print('file :', file)
print('random :', n)
print('RMSE :', rmse)
print('R2 :', r2)

# file : new_test_1.csv
# random : 67311
# RMSE : 97.30294201117887
# R2 : 0.39731011847620046

# file : new_test_2.csv 
# RMSE : 98.5586941509583
# R2 : 0.40439780745188336