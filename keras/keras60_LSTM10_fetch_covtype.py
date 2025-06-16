from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout, BatchNormalization,Flatten, MaxPooling2D
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

x = x.reshape(x.shape[0], 9, 6)
y = y.reshape(y.shape[0], 7)
    
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50, stratify=y)

model = Sequential()
model.add(LSTM(30, input_shape=(9, 6)))
model.add(Flatten())
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=2,
                   restore_best_weights=True)
str = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.2,
          verbose=2)
end = time.time()


results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
print('[BCE] :', round(results[0], 4))
print('[ACC] :', round(results[1], 4))
print('걸린시간: ', end-str)

