#분류 방식(RMSE) 낮을수록 좋음`-` 킹왕짱 1.0%
# 20250611 보강

# 1. R2를 음수가 아닌 0.5 이하로 만들것
# 2. 데이터는 건들지 말것
# 3. 레이어는 인풋 아웃풋 포함 7개 이상
# 4. batch_size=1
# 5. 히든레이어의 노드는 10개 이상 100개 이하
# 6. train 사이즈 75%
# 7. epoch 100번이상
# 8. loss 지표는 mse, mae

import numpy as np  
import matplotlib.pyplot as plt   
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# y = np.array([1,2,4,3,5,7,9,3,8,14,17,18,16,13,14,20,12,15,16,17])

x_train, x_test , y_train, y_test  = train_test_split(
                                    x,y,
                                    #train_size=0.7
                                    shuffle=True, #디폴트 true
                                    test_size=0.35,
                                    random_state=1)

#2. 모델 
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

#3. 피트
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, verbose=2, batch_size=1)
#4. 학습
loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

print("loss:", loss)
print("x의 대한 예측값:", results)

from sklearn.metrics import r2_score, mean_squared_error 

# def RMSE (y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# rmse = RMSE(y_test, results)
# print('RMSE:', rmse)

r2 = r2_score(y_test, results)

print("r2 스코어:", r2)

# r2 스코어: 0.3131472307015777

# r2 스코어: 0.03485262117378285