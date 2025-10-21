import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
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

#2. 모델구성 13
input = Input(shape=(13,))
dense1 = Dense(10)(input)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(10)(drop1)
dense3 = Dense(15)(dense2)
drop2 = Dropout(0.1)(dense3)
dense4 = Dense(10)(drop2)
output = Dense(1)(dense4)
model = Model(inputs= input, outputs= output)
model.summary()
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         [(None, 13)]              0
# dense (Dense)                (None, 10)                140
# dropout (Dropout)            (None, 10)                0
# dense_1 (Dense)              (None, 10)                110
# dense_2 (Dense)              (None, 15)                165
# dropout_1 (Dropout)          (None, 15)                0
# dense_3 (Dense)              (None, 10)                160
# dense_4 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 586
# Trainable params: 586
# Non-trainable params: 0

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)

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