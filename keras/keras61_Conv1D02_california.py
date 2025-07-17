import ssl
# import certifi

ssl._create_default_https_context = ssl._create_unverified_context


import sklearn as sk
print(sk.__version__)       # 1.1.3
import tensorflow as tf
print(tf.__version__)       # 2.9.3
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
#1. 데이터
dataset = fetch_california_housing()
print(dataset)          # y 데이터는 타겟데이터
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(x.shape)      # (20640, 8)
print(y)
print(y.shape)      # (20640,)

# [실습]  r2 > 0.59

x = x.reshape(20640, 8, 1)
y = y.reshape(20640, 1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=111)

model = Sequential()

model.add(Conv1D(20, kernel_size=2, input_shape=(8,1)))
model.add(Conv1D(19,7))
model.add(Flatten())
model.add(Dense(150))
model.add(Dense(75))
model.add(Dense(25))
model.add(Dense(1))

str = time.time()
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=100)
end = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, results)

print('r2 score: ', r2)
print('걸린시간 :', end - str)

# r2 score:  0.6823191790326089
# 걸린시간 : 720.910849571228

# r2 score:  0.43209223084582526
# 걸린시간 : 48.953747510910034