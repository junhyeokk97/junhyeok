import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import time

from sklearn.datasets import fetch_california_housing
# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=18,
    shuffle=True)

print(x_train.shape, y_train.shape) # (18576, 8) (18576,)
print(x_test.shape, y_test.shape) # (2064, 8) (2064,)

x_train = x_train.reshape(18576, 4, 2, 1)
x_test = x_test.reshape(2064, 4, 2, 1)
print(x_train.shape, x_test.shape)

y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
print(y_train.shape, y_test.shape)


model = Sequential()
model.add(Conv2D(32, (1,1), strides=1, input_shape=(4, 2, 1),padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(1,1),padding='same', activation='relu'))
model.add(Conv2D(16, (1,1),padding='same' , activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=2,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
          verbose=2)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('loss: ', loss[0])
print('acc: ', loss[1])

import matplotlib.pyplot as plt
plt.imshow(x_train, 'gray')
plt.show()