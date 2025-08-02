import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

# [실습] 리스트의 train_test_split 10:3:3으로 나눈다.
 
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.85,
    shuffle=False,
    random_state=12)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,y_train,
    train_size=0.8,
    shuffle=False,
    random_state=12)

print(x_train)  # [ 1  2  3  4  5  6  7  8  9 10]
print(x_val) # [11 12 13] /// >> traindp 포함된다는 것 인지하기.
print(x_test)   # [14 15 16]

print("###")
print(y_train)
print(y_val)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
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









