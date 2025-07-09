from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import sklearn as sk    #print(sk.__version__)   #1.6.1 -> 1.1.3
import numpy as np      #print(np.__version__)   #1.23.0
import time
from sklearn.datasets import load_boston
import time
#1. 데이터
dataset = load_boston()

x = dataset.data
y = dataset.target

# print(x)
# print(x.shape)  #(506, 13)
# print(y)
# print(y.shape)  #(506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=243)

#2. 모델 구성
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, input_shape=(13,1)))
# model.add(Dense(50, input_dim = 13))
model.add(Conv1D(31,12))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1))
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
str = time.time()
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
print('걸린시간: ', end-str)
# RMSE : 5.750288203369232
# R2 : 0.6550242566452449


# RMSE : 5.105558705672148
# R2 : 0.7280457290614397
# 걸린시간:  94.20147109031677