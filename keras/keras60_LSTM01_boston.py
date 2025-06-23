from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import sklearn as sk    #print(sk.__version__)   #1.6.1 -> 1.1.3
import numpy as np      #print(np.__version__)   #1.23.0
import time
from sklearn.datasets import load_boston
#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

# print(x)
# print(x.shape)  #(506, 13)
# print(y)
# print(y.shape)  #(506,)
x = x.reshape(506, 13, 1)
y = y.reshape(506, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=243)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

str = time.time()
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)
print('###################')
print('RMSE :', rmse)   
print('R2 :', r2)
print('걸린시간 :', end - str)

# RMSE : 3.9994100494655958
# R2 : 0.8331211869044535
# 걸린시간 : 69.50155353546143