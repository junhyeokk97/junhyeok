from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

path = './_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test_1.csv', )
# new_test_csv = pd.read_csv(path + 'new_test_2.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop(['count'], axis=1)
# print(x)    #[10886 rows x 10 columns]
y = train_csv['count']
# print(y)    #(10886,)

import random as rd
n = rd.randint(1, 10000)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=n)

# print(x_train.shape)
# print(x_test .shape)
# print(y_train.shape)
# print(y_test .shape)
# (8708, 10)
# (2178, 10)
# (8708,)
# (2178,)

model = Sequential()
model.add(Dense(50, input_dim=10, activation='relu'))
model.add(Dense(70, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=32)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
# print(x_test)   #[2178 rows x 10 columns]
# exit()
def RMSE(a, b) : return np.sqrt(mean_squared_error(a, b))
rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)
# print(new_test_csv.shape)   #(6493, 10)
# exit()
y_submit = model.predict(new_test_csv)
submission_csv['count'] = y_submit
file = 'submission_1.csv'
submission_csv.to_csv(path + file, index=False)

print('file :', file)
print('random :', n)
print('RMSE :', rmse)
print('R2 :', r2)
