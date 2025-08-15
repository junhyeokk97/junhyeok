import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.layers import GlobalAvgPool2D
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
model.add(Conv2D(100, (2,2), strides=2, input_shape=(10, 10, 1)))   # 8,8,64
model.add(Conv2D(filters=50, kernel_size=(2,2)))        # 6,6,32
model.add(Conv2D(30, (2,2), padding='same'))                             # 4,4,16
# model.add(Flatten())
model.add(GlobalAvgPool2D())
# model.add(Dense(units=16))
# model.add(Dense(units=16))
model.add(Dense(units=10, activation='softmax'))
model.summary()


##### Flatten #####
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 5, 5, 100)         500
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 4, 4, 50)          20050
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 4, 30)          6030
# _________________________________________________________________
# flatten (Flatten)            (None, 480)               0
# _________________________________________________________________
# dense (Dense)                (None, 10)                4810
# =================================================================
# Total params: 31,390
# Trainable params: 31,390
# Non-trainable params: 0


##### Global #####
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 5, 5, 100)         500
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 4, 4, 50)          20050
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 4, 30)          6030
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 30)                0
# _________________________________________________________________
# dense (Dense)                (None, 10)                310
# =================================================================
# Total params: 26,890
# Trainable params: 26,890
# Non-trainable params: 0