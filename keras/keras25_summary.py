from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(1))

model.summary()
#dense (Dense)               (None, 3)                 6

#dense_1 (Dense)             (None, 4)                 16

#dense_2 (Dense)             (None, 2)                 10

#dense_3 (Dense)             (None, 1)                 3
# Non-trainable params: 0 => 사전학습된 모델을 사용해 이미 가중치가 나와있어 훈련이 필요없는 상태