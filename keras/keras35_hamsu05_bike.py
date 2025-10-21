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
path = './_data/kaggle/bike'
train_csv = pd.read_csv(path + '/train.csv', index_col=0)
test_csv =  pd.read_csv(path + '/test.csv', index_col=0)
submission_csv = pd.read_csv(path + '/sampleSubmission.csv')

x = train_csv.drop(columns=['casual','registered','count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100)

# 3. 모델 구성 8
input = Input(shape=(8,))
dense1 = Dense(10)(input)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(10)(drop1)
dense3 = Dense(7)(dense2)
dense4 = Dense(4)(dense3)
output = Dense(1)(dense4)
model = Model(inputs = input, outputs = output)
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 8)]               0
#  dense (Dense)               (None, 10)                90
#  dropout (Dropout)           (None, 10)                0
#  dense_1 (Dense)             (None, 10)                110
#  dense_2 (Dense)             (None, 7)                 77
#  dense_3 (Dense)             (None, 4)                 32
#  dense_4 (Dense)             (None, 1)                 5
# =================================================================
# Total params: 314
# Trainable params: 314
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
history = model.fit(x_train, y_train, epochs=1000, verbose=2, batch_size=200, validation_split=0.1)


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
Loss: 21361.49609375
RMSE : 146.15572549082708
'''

# Loss: 20956.59375
# RMSE : 144.7639230087911
# 걸린시간:  16.91638684272766
# GPU 있다.

# Loss: 20692.140625
# RMSE : 143.84762407312598
# 걸린시간:  13.863747119903564
# GPU 없다.


# Dropout
# Loss: 24343.873046875
# RMSE : 156.02523217144773
# 걸린시간:  238.7018644809723
# GPU 있다.