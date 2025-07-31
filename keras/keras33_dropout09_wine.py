import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(datasets.feature_names)

print(x.shape)  # (178, 13)
print(y.shape)  # (178,)
print(np.unique(y, return_counts=True))

ohe = OneHotEncoder(sparse=False)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
print(type(x))  # <class 'numpy.ndarray'>
print(type(y))  # <class 'numpy.ndarray'>


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337, stratify=y # stratify는 x와 y를 균등한 비율로 분배
)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(75, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(56, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)


# import datetime
# date = datetime.datetime.now()
# print(date)                     
# print(type(date))               
# date = date.strftime('%m%d_%H%M')
# print(date)                    
# print(type(date))              

# path1 = './_save/keras28_mcp/09_wine/'
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
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
print('f1_score : ', f1)

# loss :  0.10895264893770218
# acc :  0.9444444179534912
# f1_score :  0.945824706694272

import tensorflow as tf

end = time.time()

print("걸린시간: ", end - start)

gpus = tf.config.list_physical_devices('GPU')           # tensorflow 2.7.4 / 2.9.0 = cpu버전

if gpus:
    print('GPU 있다.')
else:
    print('GPU 없다.')
    

# loss :  0.24891138076782227
# acc :  0.8055555820465088
# f1_score :  0.8077294685990338
# 걸린시간:  5.666504144668579
# GPU 있다.

# loss :  0.3273237347602844
# acc :  0.6388888955116272
# f1_score :  0.6328395061728395
# 걸린시간:  3.18302583694458
# GPU 없다.