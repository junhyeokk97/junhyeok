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
# model.add(SimpleRNN(10, input_shape=(3,2)))     # 시계열 데이터는 y값을 구하기 위해서 사용하지만, 
# model.add(SimpleRNN(units=16, input_shape=(3,1))) # units이 들어가면 자리가 바뀌면 안된다.
model.add(LSTM(10, input_shape=(3,1)))
# model.add(GRU(units=16, input_shape=(3,1)))
model.add(Dense(5, activation='relu'))            # 다차원 데이터라도 2차원 데이터와 연결된다.
model.add(Dense(1))
model.summary()
# Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120
#  dense (Dense)               (None, 5)                 55
#  dense_1 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0

# 파라미터 개수 = (units * feature) + (units * uints) + (bias * units)
#              =    (1   *   10)  +   (10  *   10)  +   (1   *   10)
#              =   units * ( feature + units + bias)
#              =    10   *       (1 + 10 + 1)   =   120

model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3,2)))     # 시계열 데이터는 y값을 구하기 위해서 사용하지만, 
# model.add(SimpleRNN(units=16, input_shape=(3,1))) # units이 들어가면 자리가 바뀌면 안된다.
model.add(LSTM(10, input_shape=(3,1)))
# model.add(GRU(units=16, input_shape=(3,1)))
model.add(Dense(5, activation='relu'))            # 다차원 데이터라도 2차원 데이터와 연결된다.
model.add(Dense(1))
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm_1 (LSTM)               (None, 10)                480
#  dense_2 (Dense)             (None, 5)                 55
#  dense_3 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 541
# Trainable params: 541
# Non-trainable params: 0

model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3,2)))     # 시계열 데이터는 y값을 구하기 위해서 사용하지만, 
# model.add(SimpleRNN(units=16, input_shape=(3,1))) # units이 들어가면 자리가 바뀌면 안된다.
# model.add(LSTM(10, input_shape=(3,1)))
model.add(GRU(units=10, input_shape=(3,1)))
model.add(Dense(5, activation='relu'))            # 다차원 데이터라도 2차원 데이터와 연결된다.
model.add(Dense(1))
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  gru (GRU)                   (None, 10)                390        > bias를 두 개로 잡는다.
#  dense_4 (Dense)             (None, 5)                 55
#  dense_5 (Dense)             (None, 1)                 6
# =================================================================
# Total params: 451
# Trainable params: 451
# Non-trainable params: 0