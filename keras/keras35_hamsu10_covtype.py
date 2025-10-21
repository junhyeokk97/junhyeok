import ssl

# SSL 인증서 문제 해결
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
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

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
input = Input(shape=(54,))
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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss', mode='min', patience=10, restore_best_weights=True
)

model.fit(x_train, y_train, epochs=10000, batch_size=1500,
          validation_split=0.2, callbacks=[es], verbose=2)

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

# loss :  0.31944671273231506
# acc :  0.8706143498420715
# f1 :  0.7842552885163677