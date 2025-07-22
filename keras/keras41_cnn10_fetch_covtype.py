from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization,Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

y = y.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(y)
# print(y)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50, stratify=y)

print(x_train.shape, x_test.shape)  # (464809, 54) (116203, 54)
x_train = x_train.reshape(464809, 9, 6, 1)
x_test = x_test.reshape(116203, 9, 6, 1)
print(x_train.shape, x_test.shape)

model = Sequential()
model.add(Conv2D(32, (3,3), strides=1, input_shape=(9, 6, 1),padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(2,2),padding='same', activation='relu'))
model.add(Conv2D(16, (2,2),padding='same' , activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=2,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.2,
          verbose=2)

results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
print('[BCE] :', round(results[0], 4))
print('[ACC] :', round(results[1], 4))