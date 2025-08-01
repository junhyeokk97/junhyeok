import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] 리스트의 슬라이싱으로 10:3:3으로 나눈다.
 
# x_train = x(range(1, 11))
# x_val = x(range(11, 14))
# x_test = x(range(14, 17))

x_train = x[:10]
x_val = x[10:14]
x_test = x[14:]
print(x_train)
print(x_val)
print(x_test)

y_train = y[:10]
y_val = y[10:14]
y_test = y[14:]
print(y_train)
print(y_val)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=1,
          validation_data=(x_val,y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict([17])
print('loss: ', loss)
print('[17]의 예측값: ', results)









