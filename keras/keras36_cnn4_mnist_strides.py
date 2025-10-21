import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

#  x reshape >> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)   # 10000은 0번째 shape, 28은 각각 첫 번째 두 번째,1은 세 번째 shape
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])로 출력해도 됨.
print(x_train.shape, x_test.shape)   # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# #2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, input_shape=(10, 10, 1)))   # 8,8,64
model.add(Conv2D(filters=32, kernel_size=(3,3)))        # 6,6,32
# model.add(Conv2D(16, (3,3)))                             # 4,4,16
model.add(Flatten())
model.add(Dense(units=16))
model.add(Dense(units=16))
model.add(Dense(units=10, activation='softmax'))
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 8, 8, 64)          640
#  conv2d_1 (Conv2D)           (None, 6, 6, 32)          18464
#  conv2d_2 (Conv2D)           (None, 4, 4, 16)          4624
#  flatten (Flatten)           (None, 256)               0
#  dense (Dense)               (None, 16)                4112
#  dense_1 (Dense)             (None, 16)                272
#  dense_2 (Dense)             (None, 10)                170
# =================================================================
# Total params: 28,282
# Trainable params: 28,282
# Non-trainable params: 0
