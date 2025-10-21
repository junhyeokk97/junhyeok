from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, BatchNormalization,Flatten, MaxPooling2D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

#1. 데이터
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

x = dataset.data    #(569, 30)
y = dataset.target  #(569,)

x = x.reshape(569, 6, 5)
y = y.reshape(569, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=50,)

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (455, 30) (114, 30)
# x_train = x_train.reshape(455, 30, 1)
# x_test = x_test.reshape(114, 30, 1)
# y = y.reshape(569, 1)
# print(x_train.shape, x_test.shape)  # (455, 6, 5, 1) (114, 6, 5, 1)

model = Sequential()
model.add(Conv1D(10, kernel_size=2, input_shape=(6, 5)))
model.add(Conv1D(9, 5))
model.add(Flatten())
model.add(Dense(15, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=2,
                   restore_best_weights=True)
str = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
          verbose=2)
end = time.time()


results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
print('[BCE] :', round(results[0], 4))
print('[ACC] :', round(results[1], 4))
print('걸린시간: ', end - str)

# [BCE] : 0.1001
# [ACC] : 0.9737


# lstm
# [BCE] : 0.1176
# [ACC] : 0.9298
# 걸린시간:  5.158300161361694

# conv1d
# [BCE] : 0.6499
# [ACC] : 0.6579
# 걸린시간:  15.798945426940918