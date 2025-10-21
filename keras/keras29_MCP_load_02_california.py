import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)

#2. 모델 구성
# model = Sequential()
# model.add(Dense(40, input_dim=8))
# model.add(Dense(70))
# model.add(Dense(90))
# model.add(Dense(40))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# es = EarlyStopping(
#     monitor='val_loss',
#     mode='min',
#     patience=20,
#     restore_best_weights=False,
# )

# ################# mcp 세이브 파일명 만들기 ##################
# import datetime
# date = datetime.datetime.now()
# print(date)                     
# print(type(date))               
# date = date.strftime('%m%d_%H%M')
# print(date)                    
# print(type(date))              

path = './_save/keras28_mcp/02_california/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# filepath = "".join([path, 'k28_', date, '_', filename])

# from tensorflow.keras.callbacks import ModelCheckpoint
# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto',
#     save_best_only=True, 
#     filepath=filepath
# )

# hist = model.fit(
#         x_train, y_train, epochs=100000, 
#         batch_size=32, verbose=2, validation_split=0.2,
#         callbacks=[es, mcp],
#         )

from tensorflow.keras.models import load_model
model = load_model(path+ 'k28_0604_1114_0068-0.6161.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)

print('############')
print('RMSE :', rmse)
print('R2 :', r2)

# ############
# RMSE : 0.8156702244575734
# R2 : 0.5077288616803208