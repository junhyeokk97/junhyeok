# cnn > dnn

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
import time
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# 스케일링
x_train = x_train/255.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train))  # 1.0 0.0
print(np.max(x_test), np.min(x_test))    # 1.0 0.0

# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test[2])

print(x_train.shape[0])  # 60000
print(x_train.shape[1])  # 28
print(x_train.shape[2])  # 28
# print(x_train.shape[3])  # error / x_train의 데이터는 60000, 28 28 이기때문에 세번째 자리가 없어서 에러.

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(-1, 1)   # -1은 데이터 전체의 끝 값
print(y_train.shape, y_test.shape)  # (60000, 1) (10000, 1)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델 // 성능 0.98 이상 // 시간체크(cnn떄와 시간 비교)
model = Sequential()
model.add(Dense(1024, input_dim=784, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dense(units=108, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=54, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=50,
                   restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=150, validation_split=0.2, verbose=2, callbacks=[es])
end = time.time()


loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


print('time: ', end-start)
print('loss: ', loss[0])
print('acc: ', loss[1])

# time:  269.46069622039795
# loss:  0.14690028131008148
# acc:  0.9850999712944031