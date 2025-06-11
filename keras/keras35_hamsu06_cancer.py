from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

#1. 데이터
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

x = dataset.data    #(569, 30)
y = dataset.target  #(569,)


r = 42 #random.randint(1, 10000)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=r,
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

#2. 모델구성 30
input = Input(shape=(30,))
dense1 = Dense(100)(input)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(100)(drop1)
drop2 = Dropout(0.1)(dense2)
dense3 = Dense(50)(drop2)
drop3 = Dropout(0.1)(dense2)
dense4 = Dense(10)(drop3)
output = Dense(1)(dense4)
model = Model(inputs= input, outputs= output)
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 30)]              0
#  dense (Dense)               (None, 100)               3100
#  dropout (Dropout)           (None, 100)               0
#  dense_1 (Dense)             (None, 100)               10100
#  dropout_2 (Dropout)         (None, 100)               0
#  dense_3 (Dense)             (None, 10)                1010
#  dense_4 (Dense)             (None, 1)                 11
# =================================================================
# Total params: 14,221
# Trainable params: 14,221
# Non-trainable params: 0


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')               # mse는 거리를 기준으로 하는데 sigmoid에는 의미없는 거리일 뿐임.
model.compile(loss='binary_crossentropy', optimizer='adam', # 이진 분류에는 100% binary_crossentropy를 쓴다.
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True
)

start_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=100000, batch_size=32,
    verbose=2, validation_split=0.2,
    callbacks=[es,],
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# print(y_predict[:10])
y_predict = np.round(y_predict) # pyhon의 반올림

print('[BCE](소숫점 4번째 자리까지 표시) :', round(results[0], 4))      # [BCE](소숫점 4번째 자리까지 표시) : 0.1399
print('[ACC](소숫점 4번째 자리까지 표시) :', round(results[1], 4))      # [ACC](소숫점 4번째 자리까지 표시) : 0.9415   

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
accuracy_score = np.round(accuracy_score, 4)
print("acc_score :", accuracy_score)
# print('걸린 시간 :', round(end_time - start_time, 2), '초')

# [BCE](소숫점 4번째 자리까지 표시) : 0.0978
# [ACC](소숫점 4번째 자리까지 표시) : 0.9766
# acc_score : 0.9766
# 걸린 시간 : 4.66 초


# MinMaxScaler
# [BCE](소숫점 4번째 자리까지 표시) : 0.0705
# [ACC](소숫점 4번째 자리까지 표시) : 0.9766
# acc_score : 0.9766

# MaxAbsScaler
# [BCE](소숫점 4번째 자리까지 표시) : 0.0671
# [ACC](소숫점 4번째 자리까지 표시) : 0.9649
# acc_score : 0.9649

# StandardScaler
# [BCE](소숫점 4번째 자리까지 표시) : 0.1306
# [ACC](소숫점 4번째 자리까지 표시) : 0.9825
# acc_score : 0.9825

# RobustScaler
# [BCE](소숫점 4번째 자리까지 표시) : 0.0761
# [ACC](소숫점 4번째 자리까지 표시) : 0.9766
# acc_score : 0.9766