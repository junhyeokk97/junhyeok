import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
import time
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

# print(x.shape)  #(506, 13)
# print(y.shape)  #(506,)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True,
    random_state=333,)

print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)

x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)
print(x_train.shape, x_test.shape)  # (404, 13, 1, 1) (102, 13, 1, 1)

# y_train = pd.get_dummies(y_train.reshape(-1))
# y_test = pd.get_dummies(y_test.reshape(-1))
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
print(y_train.shape, y_test.shape)  # (404,) (102,)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(32, (1,1), strides=1, input_shape=(13, 1, 1),padding='valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(1,1),padding='same', activation='relu'))
model.add(Conv2D(16, (1,1),padding='same' , activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()
exit()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('###################')
print('loss: ', loss)   
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)


print('r2: ', r2)     

# 여기서는 이게 돌아갈리가 없음. x_test와 results의 열의 수가 다름
# import matplotlib.pyplot as plt
# plt.scatter(x_test, y_test)
# plt.plot(x_test, results, color='red')
# plt.show()

# loss:  24.057729721069336
# r2:  0.7547106028027055