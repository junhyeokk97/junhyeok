from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(11, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(13, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
early = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
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
    batch_size=100,
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

# Loss: 0.6769302487373352
# RMSE : 0.8227580320289799
# 걸린시간:  18.33233642578125
# GPU 없다.

# Loss: 0.6439374089241028
# RMSE : 0.8024570331742468
# 걸린시간:  23.56201696395874
# GPU 있다.