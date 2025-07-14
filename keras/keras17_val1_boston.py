"""
tf274 - keras  // anaconda 통설치

conda create -n tf274 python==3.9.7 andnaconda
tensorflow 2.7.4
numpy 1.20.3
scikit 0.24.2
matplotlib 3.4.3
pandas 1.3.4
"""
import sklearn as sk
print(sk.__version__) 

from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets = load_boston()
print(datasets)
print(datasets.DESCR) #(506,13))
print(datasets.feature_names)
x = datasets.data
y = datasets.target

print(x)
print(x.shape) #(506, 13)
print(y)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=814)
    
#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x,y, epochs=500, batch_size=1, validation_split=0.2)

#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict([x_test])

print("loss : ", loss)
print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)
# 기존
# loss :   29.726648330688477
# r2 스코어 : 0.7584895173069439

# validation 적용
# loss :  16.460010528564453
# r2 스코어 :  0.8662726517591319