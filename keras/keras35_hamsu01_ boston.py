# 27-3 카피
import tensorflow as tf
import sklearn as sk
print(sk.__version__)       #0.24.2

import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

#1. 데이터
from sklearn.datasets import load_boston
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True,
    random_state=333,
)

mms = MinMaxScaler()
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)

#2. 모델 구성
input1 = Input(shape=(13,))
dense1 = Dense(10, name='mm1')(input1)
dense2 = Dense(12)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(15)(drop1)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(7)(dense2)
dense5 = Dense(3)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs= input1, outputs= output1)
model.summary()
# Layer (type)                 Output Shape              Param #
# ================================================================
# input_1 (InputLayer)         [(None, 13)]              0______________________________________________________________
# mm1 (Dense)                  (None, 10)                140_____________________________________________________________
# dense (Dense)                (None, 12)                132_______________________________________________________________
# dense_2 (Dense)              (None, 7)                 91___________________________________________________________
# dense_3 (Dense)              (None, 3)                 24________________________________________________________________
# dense_4 (Dense)              (None, 1)                 4
# =================================================================
# Total params: 391
# Trainable params: 391
# Non-trainable params: 0

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss', mode='min', patience=20,
    restore_best_weights=True,
)

start = time.time()
hist = model.fit(
    x_train, y_train, 
    epochs=100, batch_size=32, 
    verbose=2, validation_split=0.2,
    callbacks=[es],
)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

end = time.time()

print("걸린시간: ", end - start)

gpus = tf.config.list_physical_devices('GPU') 

if gpus:
    print('GPU 있다.')
else:
    print('GPU 없다.')

print('RMSE :', rmse)   
print('R2 :', r2)

# 걸린시간:  7.150681495666504
# GPU 있다.
# RMSE : 5.723329805177917
# R2 : 0.666019062390324