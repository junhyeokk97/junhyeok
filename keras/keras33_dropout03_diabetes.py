from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import time
import os

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 1. 데이터
dataset =  load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=555)

# 2. 모델 구성
model = Sequential()
model.add(Dense(25, activation='relu', input_dim=10))
model.add(Dropout(0.2))
model.add(Dense(75, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(36, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(18, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
# os.makedirs('./_save/keras28_mcp/03_diabetes/', exist_ok=True)
# path = './_save/keras28_mcp/03_diabetes/'
# filename = '03_diabetes_{epoch:04d}_{val_loss:.4f}.hdf5'

# early = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=30,
#     restore_best_weights=True,
# )


model.compile(loss='mse', optimizer ='adam')
start = time.time()
model.fit(x_test, y_test, epochs=100, batch_size=32, validation_split=0.2)


# path = './_save/keras28_mcp/03_diabetes/'
# filename = '03_diabetes_{epoch:04d}_{val_loss:.4f}.hdf5'

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
Loss: 2067.351318359375
RMSE : 45.468134227907235
'''

# Loss: 1859.0196533203125
# RMSE : 43.11635035324073
# 걸린시간:  4.621135234832764
# GPU 있다.

# Loss: 1694.9322509765625
# RMSE : 41.16955188606869
# 걸린시간:  3.379331111907959
# GPU 없다.

# Loss: 3825.2607421875
# RMSE : 61.84869131638113
# 걸린시간:  29.992871046066284
# GPU 있다.