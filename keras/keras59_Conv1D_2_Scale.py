# 53 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Conv1D, Flatten

#1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(x.shape[0], x.shape[1],1)
print(x.shape)  # (13, 3, 1)

model = Sequential()
model.add(Conv1D(24, kernel_size=2, input_shape=(3,1)))    # (None, 3, 24)
model.add(Conv1D(20, 2))                     # (None, 2, 20)
model.add(Flatten())
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=500, verbose=2)

x_predict = x_predict.reshape(1, 3, 1)
loss = model.evaluate(x, y)
results = model.predict(x_predict)

print('loss: ', loss)
print('results: ', results)

# loss:  1.9441373348236084
# results:  [[84.82594]]