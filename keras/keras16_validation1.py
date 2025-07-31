#08-1 카피

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) # (10,)
# print(y.shape) # (10,)

x_train = np.array([1,2,3,4,5,6,])
y_train = np.array([1,2,3,4,5,6,])

x_val = np.array([7,8,])                # validation까지는 train data
y_val = np.array([7,8,])

x_test = np.array([9,10])
y_test = np.array([9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 ,훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=50, batch_size=1,
          validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([11])

print('loss : ', loss)
print('[11]의 예측값 : ', results)

# 결과
# loss :  1.5916157281026244e-12
# [11]의 예측값 :  [[11.]]
