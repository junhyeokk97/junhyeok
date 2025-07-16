from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout,Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import time

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=18,
    shuffle=True
)

# 2. 모델 구성
input = Input(shape=(8,))
dense1 = Dense(10, name='zz1')(input)
dense2 = Dense(15)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(9)(drop1)
drop2 = Dropout(0.1)(dense3)
dense4 = Dense(4)(drop2)
output = Dense(1)(dense4)
model = Model(inputs = input, outputs = output)
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 8)]               0
#  zz1 (Dense)                 (None, 10)                90
#  dense (Dense)               (None, 15)                165
#  dropout (Dropout)           (None, 15)                0
#  dense_1 (Dense)             (None, 9)                 144
#  dropout_1 (Dropout)         (None, 9)                 0
#  dense_2 (Dense)             (None, 4)                 40
#  dense_3 (Dense)             (None, 1)                 5
# =================================================================
# Total params: 444
# Trainable params: 444
# Non-trainable params: 0

# 3. 컴파일, 훈련
early = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)

# path = './_save/keras28_mcp/02_california/'
# filename = '02_california_{epoch:04d}_{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     filepath=path+filename
# )

model.compile(loss='mse', optimizer='adam')

start = time.time()
hist = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    verbose=2,
    validation_split=0.1,
    callbacks=[early],    
)


# 4. 평가, 예측
results = model.predict(x_test)
loss = model.evaluate(x_test, y_test, verbose=0)
rmse = RMSE(results, y_test)
print("Loss:", loss)
print("RMSE :", rmse)

import tensorflow as tf

end = time.time()

print("걸린시간: ", end - start)

gpus = tf.config.list_physical_devices('GPU')           # tensorflow 2.7.4 / 2.9.0 = cpu버전

if gpus:
    print('GPU 있다.')
else:
    print('GPU 없다.')

'''
Loss: 0.599709689617157
RMSE : 0.7744092808261633
'''

# Loss: 0.6976410150527954
# RMSE : 0.8352490515537333
# 걸린시간:  178.38141107559204
# GPU 있다.

# Loss: 0.6769302487373352
# RMSE : 0.8227580320289799
# 걸린시간:  18.33233642578125
# GPU 없다.