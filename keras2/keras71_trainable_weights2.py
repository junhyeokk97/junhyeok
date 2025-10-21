import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import random

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3,4,5])
x = np.array([1])
y = np.array([1])


model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

################### 동결 ###################
# model.trainable = False     # 동결
model.trainable = True      # 동결x  >>  defualt


model.compile(loss='mse', optimizer='adam')
model.fit(x,y, batch_size=1, epochs=100, verbose=0)

print("======================")
print(model.weights)
print("======================")

y_pred = model.predict(x)
print(y_pred)

# 동결                              h2_1 -0.5494663863483774
# [[0.45656443]                     h2_2 -0.411306976002892
#  [0.91312885]
#  [1.369693  ]
#  [1.8262577 ]
#  [2.2828221 ]]

# 동결 x 
# [[1.2034489]
#  [2.1232457]
#  [3.043042 ]
#  [3.9628391]
#  [4.882636 ]]

# x=1, y=1 / 가중치 동결 후 훈련.
# [[0.45656443]]
################# 동결 손계산

# 동결 하지 않은 것 훈련 계산
# [[1.005533]]