# 26-3 카피

import sklearn as sk
print(sk.__version__)       #0.24.2

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

#1. 데이터
from sklearn.datasets import load_boston
datasets = load_boston()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

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

print(np.min(x_train), np.max(x_train)) # 0.0 / 1.0000000000000002 -> 부동 소수점 연산을 이진법으로 하다보니 오차가 생긴 것
print(np.min(x_test), np.max(x_test))   # -0.00557837618540494 / 1.1478180091225068

# a = 0.1 / b = 0.2
# print(a+b)  # 0.30000000000000004 -> 부동 소수점 연산

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(1))

model.summary()

path = './_save/keras26/'
# model.save(path + 'keras26_1_save.h5')
model.save_weights(path + 'keras26_5_save1.h5') # 가중치값'만' 저장 (모델이 커지면 커질수록 save() 파일과의 용량 차이가 심해짐)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(
        monitor='val_loss', mode = 'min', patience=20,
        restore_best_weights=False,
)

model.fit(
    x_train, y_train, 
    epochs=1000000, batch_size=1, 
    verbose=2, validation_split=0.2,
    callbacks=[es],
)

# path = './_save/keras26/'
# model.save(path + 'keras26_3_save.h5')
model.save_weights(path + 'keras26_5_save2.h5') # 가중치값'만' 저장 (모델이 커지면 커질수록 save() 파일과의 용량 차이가 심해짐)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('###################')
print('RMSE :', rmse)   
print('R2 :', r2)

# ###################
# RMSE : 4.747429603990841
# R2 : 0.7702046551092533