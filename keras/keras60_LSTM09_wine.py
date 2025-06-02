from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM,  BatchNormalization,Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.datasets import load_wine

datasets = load_wine()
x = datasets.data
y = datasets.target

print(datasets.feature_names)

print(x.shape)  # (178, 13)
print(y.shape)  # (178,)
print(np.unique(y, return_counts=True))

# ohe = OneHotEncoder(sparse=False)
# y = y.reshape(-1, 1)
# y = ohe.fit_transform(y)
# print(type(x))  # <class 'numpy.ndarray'>
# print(type(y))  # <class 'numpy.ndarray'>

x = x.reshape(x.shape[0], x.shape[1], 1)
y = y.reshape(y.shape[0], 1)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337, stratify=y # stratify는 x와 y를 균등한 비율로 분배
)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)  # (142, 13) (36, 13)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])

model = Sequential()
model.add(LSTM(8, input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

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
print('걸린 시간 :', end - str)


# [BCE] : 0.2097
# [ACC] : 0.9444