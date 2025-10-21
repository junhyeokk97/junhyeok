import ssl

# SSL 인증서 문제 해결
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import numpy as np
import pandas as pd

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
    x, y, test_size=0.2, random_state=50, stratify=y
)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=54, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(
#     monitor='val_loss', mode='min', patience=10, restore_best_weights=True
# )
# import datetime
# date = datetime.datetime.now()
# # print(date)                     
# # print(type(date))               
# date = date.strftime('%m%d_%H%M')
# # print(date)                    
# # print(type(date))              

# path1 = './_save/keras28_mcp/10_fetch_covtype/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# filepath = "".join([path1, 'k28_', date, '_', filename])

# from tensorflow.keras.callbacks import ModelCheckpoint
# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto',
#     save_best_only=True, 
#     filepath=filepath
# )
import time
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=100,
          validation_split=0.2, verbose=2)


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_predict = model.predict(x_test)
y_round = np.round(y_predict)
f1 = f1_score(y_test, y_round, average='macro')
print('f1 : ', f1)


# loss :  0.16358691453933716
# acc :  0.9360687732696533
# f1 :  0.9026860016864566

import tensorflow as tf

end = time.time()

print("걸린시간: ", end - start)

gpus = tf.config.list_physical_devices('GPU')           # tensorflow 2.7.4 / 2.9.0 = cpu버전

if gpus:
    print('GPU 있다.')
else:
    print('GPU 없다.')
    
# loss :  0.2731139063835144
# acc :  0.8875588178634644
# f1 :  0.8318334731140001
# 걸린시간:  171.4919307231903
# GPU 있다.

# loss :  0.25519701838493347
# acc :  0.8963882327079773
# f1 :  0.8334305290139269
# 걸린시간:  51.04671788215637
# GPU 없다.