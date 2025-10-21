from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([range(10), range(21, 31), range(201, 211)]) # (3, 10) > (10, 3)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [10,9,8,7,6,5,4,3,2,1],
              [9,8,7,6,5,4,3,2,1,0]]) # (3, 10) > (10, 3)

x = x.T
y = y.T

print(x.shape)
print(y.shape)

model = Sequential()
model.add(Dense(9, input_dim=3))
model.add(Dense(16))
model.add(Dense(38))
model.add(Dense(21))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') #mse
model.fit(x,y, epochs=300, batch_size=1)

loss = model.evaluate(x,y)
results = model.predict([[10, 31, 211]])
print('loss: ', loss)
print('[[10, 31, 211]]의 예측값: ', results)

# loss:  7.5555548667907715
# [[10, 31, 211]]의 예측값:  [[3.3333213]]

