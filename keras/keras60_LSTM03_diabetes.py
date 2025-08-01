import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes                               #당뇨병 diabetes
from sklearn.metrics import r2_score, mean_squared_error
import time
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (442, 10) (442,)

x = x.reshape(442, 10, 1)
y = y.reshape(442, 1)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.1,
    random_state=813
)

#2. 모델구성
model = Sequential()
model.add(LSTM(20, input_shape=(10, 1)))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(1))

str = time.time()
#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)
end = time.time()

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
results = model.predict([x_test])

print('loss : ', loss)
# print("[x_test]의 예측값 : ", results)

r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)
print('걸린시간 :', end - str)


# 목표 r2 0.62 이상
# loss :  2281.981201171875
# r2 스코어 :  0.6295709504734388 epchos=1000   layer를 원상태유지



# loss :  2930.2099609375
# r2 스코어 :  0.524345355087189
# 걸린시간 : 90.2539472579956