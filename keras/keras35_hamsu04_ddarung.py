 #https://dacon.io/competitions/open/235576/overview/description

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import time

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 1. 데이터
path = './_data/dacon/따릉이/'         
train_csv = pd.read_csv(path +'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col = 0)

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test =  train_test_split (x, y, test_size = 0.1, random_state=0)

# 2. 모델 구성 9
input = Input(shape=(9,))
dense1 = Dense(10)(input)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(15)(drop1)
dense3 = Dense(20)(dense2)
drop2 = Dropout(0.1)(dense2)
dense4 = Dense(7)(drop2)
output = Dense(1)(dense4)
model = Model(inputs = input, outputs= output)
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 9)]               0
#  dense (Dense)               (None, 10)                100
#  dropout (Dropout)           (None, 10)                0
#  dense_1 (Dense)             (None, 15)                165
#  dropout_1 (Dropout)         (None, 15)                0
#  dense_3 (Dense)             (None, 7)                 112
#  dense_4 (Dense)             (None, 1)                 8
# =================================================================
# Total params: 385
# Trainable params: 385
# Non-trainable params: 0


# 3. 컴파일, 훈련
early = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True,
)

model.compile(loss='mse', optimizer ='adam')
start = time.time()
history = model.fit(x_test, y_test, epochs=100, batch_size=32, verbose=2, validation_split=0.2)



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
Loss: 4302.62744140625
RMSE : 65.59441786781107
'''

# Loss: 2888.0380859375
# RMSE : 53.740469388440296
# 걸린시간:  5.236091375350952
# GPU 있다.

# Loss: 2843.769775390625
# RMSE : 53.327006797420744
# 걸린시간:  4.20763897895813
# GPU 없다.