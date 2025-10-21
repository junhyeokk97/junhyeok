from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

# 원본은 N,5,5,1 이미지   / N은 데이터의 수
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(5,5,1))) # (2,2)는 커널사이즈 >> 2,2 사이즈로 잘라준다. # (4, 4, 10)   2 * 2 * 1 + 1 ) * 10 = 50
model.add(Conv2D(5, (2,2)))     # (3, 3, 5)         2 * 2 * 10 + 1 ) * 5 = 205       
model.add(Conv2D(10, (3,3)))    # (1, 1, 10)        3 * 3 * 5 + 1 ) * 10 = 460
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 4, 4, 10)          50
#  conv2d_1 (Conv2D)           (None, 3, 3, 5)           205
#  conv2d_2 (Conv2D)           (None, 1, 1, 10)          460
# =================================================================
# Total params: 715
# Trainable params: 715
# Non-trainable params: 0
