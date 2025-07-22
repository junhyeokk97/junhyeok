import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Bidirectional

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

#2. 모델구성
model = Sequential()
# model.add(GRU(units=10, input_shape=(3,1)))
model.add(Bidirectional(GRU(units=10), input_shape=(3,1)))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))
model.summary()
'''


RNN : 205
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 10)                120

 dense (Dense)               (None, 7)                 77

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 205
Trainable params: 205
Non-trainable params: 0
-------------------------------------------------------------------

Bidirectional : 395
Layer (type)                Output Shape              Param #
=================================================================
 bidirectional (Bidirectiona  (None, 20)               240
 l)

 dense (Dense)               (None, 7)                 147

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 395
Trainable params: 395
Non-trainable params: 0




GRU : 475
 Layer (type)                Output Shape              Param #
=================================================================
 gru (GRU)                   (None, 10)                390

 dense (Dense)               (None, 7)                 77

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 475
Trainable params: 475
Non-trainable params: 0
-------------------------------------------------------------------

Bidirectional : 935
 Layer (type)                Output Shape              Param #
=================================================================
 bidirectional (Bidirectiona  (None, 20)               780
 l)

 dense (Dense)               (None, 7)                 147

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 935
Trainable params: 935
Non-trainable params: 0




LSTM : 565
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 10)                480
#  dense (Dense)               (None, 7)                 77
#  dense_1 (Dense)             (None, 1)                 8
# =================================================================
# Total params: 565
# Trainable params: 565
# Non-trainable params: 0
-------------------------------------------------------------------
Bidirectional : 1,115
 Layer (type)                Output Shape              Param #
=================================================================
 bidirectional (Bidirectiona  (None, 20)               960
 l)

 dense (Dense)               (None, 7)                 147

 dense_1 (Dense)             (None, 1)                 8

=================================================================
Total params: 1,115
Trainable params: 1,115
Non-trainable params: 0
'''