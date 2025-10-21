import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],  # 타임스텝 = 3
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)  # (7, 3, 1)  /  (batch_size, timesteps, feature)
# x = np.array([[[1],[2],[3]],
#               [[2],[3],[4]],
#               [[3],[4],[5]],
#               [[4],[5],[6]],
#               [[5],[6],[7]],
#               [[6],[7],[8]],
#               [[7],[8],[9]]])

model = Sequential()
# model.add(SimpleRNN(16, input_shape=(3,1)))     # 시계열 데이터는 y값을 구하기 위해서 사용하지만, 
# model.add(SimpleRNN(units=16, input_shape=(3,1))) # units이 들어가면 자리가 바뀌면 안된다. / # 다차원 데이터라도 2차원 데이터와 연결된다.
model.add(SimpleRNN(units=10, input_length=3, input_dim=1)) # (timesteps, feature)
model.add(SimpleRNN(units=10, input_dim=1, input_length=3)) # (feature, timesteps) 순서를 바꿔서 사용 가능하지만, 가독성은 떨어지는 편.


model.add(Dense(8, activation='relu'))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, )

results = model.evaluate(x, y)
print('loss: ', results)

x_pred = np.array([8,9,10]).reshape(1, 3, 1)    # (3,) >> (1, 3, 1)
y_pred = model.predict(x_pred)

print('[8,9,10]: ', y_pred)

# RNN : [8,9,10]:  [[10.999705]] 