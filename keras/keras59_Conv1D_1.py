# 52-1 copy

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Conv1D, Flatten

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
# model.add(SimpleRNN(units=16, input_shape=(3,1))) # units이 들어가면 자리가 바뀌면 안된다.
# model.add(LSTM(16, input_shape=(3,1)))
# model.add(GRU(units=16, input_shape=(3,1)))
model.add(Conv1D(filters=20, kernel_size=2,
                 padding='same',
                 input_shape=(3, 1)))    # (None, 3, 1) 상태이기 떄문에 행 무시 (3, 1)로 설정.
                                         # filters=10 >> (None, 2, 10) >> padding='same'으로 (None, 3, 10) 유지
model.add(Conv1D(19, 2))     # (None, 2 , 9)
model.add(Flatten())    # (None, 18)  # Flatten()을 사용하여 2D 데이터를 1D로 변환
model.add(Dense(4))
model.add(Dense(1))
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv1d (Conv1D)             (None, 3, 10)             30
#  conv1d_1 (Conv1D)           (None, 2, 9)              189
#  flatten (Flatten)           (None, 18)                0
#  dense (Dense)               (None, 4)                 76
#  dense_1 (Dense)             (None, 1)                 5
# =================================================================
# Total params: 300
# Trainable params: 300
# Non-trainable params: 0


model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, )

results = model.evaluate(x, y)
print('loss: ', results)

x_pred = np.array([8,9,10]).reshape(1, 3, 1)    # (3,) >> (1, 3, 1)
y_pred = model.predict(x_pred)

print('[8,9,10]: ', y_pred)

# RNN : [8,9,10]:  [[10.999705]] 