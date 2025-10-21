#https://dacon.io/competitions/open/235576/overview/description

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import time

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=100, stratify=y)

# 2. 모델 구성
model = Sequential([
    Dense(1024, activation='relu', input_dim=30),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

# 3. 컴파일, 훈련
early_stop = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=30,
    restore_best_weights=True,
)

# path = './_save/keras28_mcp/06_cancer/'
# filename = '06_cancer_{epoch:04d}_{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     save_weights_only=False,
#     filepath=path+filename
# )

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=100,
    verbose=2,
    validation_split=0.2,
)



# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
loss = results[0]
acc = results[1]
print("Loss:", loss)
print("Accuracy :", acc)

'''
Loss: 0.20353026688098907
Accuracy : 0.9298245906829834
'''

import tensorflow as tf

end = time.time()

print("걸린시간: ", end - start)

gpus = tf.config.list_physical_devices('GPU')           # tensorflow 2.7.4 / 2.9.0 = cpu버전

if gpus:
    print('GPU 있다.')
else:
    print('GPU 없다.')
    
# Loss: 0.18738189339637756
# Accuracy : 0.9122806787490845
# 걸린시간:  11.79728651046753
# GPU 있다.

# Loss: 0.13891096413135529
# Accuracy : 0.9298245906829834
# 걸린시간:  7.634494304656982
# GPU 없다.