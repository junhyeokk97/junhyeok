import ssl

ssl._create_default_https_context = ssl._create_unverified_context


import sklearn as sk
print(sk.__version__)       # 1.1.3
import tensorflow as tf
print(tf.__version__)       # 2.9.3
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

#1. 데이터
dataset = fetch_california_housing()
print(dataset)          # y 데이터는 타겟데이터
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(x.shape)      # (20640, 8)
print(y)
print(y.shape)      # (20640,)

# [실습]  r2 > 0.59

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=111)

model = Sequential()
model.add(Dense(20, input_dim=8))
model.add(Dense(500))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(50))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=0, validation_split=0.2)

print("@@@@@")
print(hist)
print("@@@@@")
print(hist.history)
print('loss@@@@')
print(hist.history['loss'])
print('val_loss@@@@')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))   # 9 x 6 사이즈 / 그래프 크기 설정.
plt.plot(hist.history['loss'], c='red', label='loss')  #  y값만 넣으면 시간 순으로 그림 / 이미지 출력하는법 다시 한 번 확인하고 암기하기
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('캘리포니아 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 우측 상단에 label 표시
plt.grid()  # 격자 표시
plt.show()

loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, results)

print('r2 score: ', r2)


# 기존
# r2 score:  0.514564716353439

# validation 적용
# r2 score:  0.4832678104157677