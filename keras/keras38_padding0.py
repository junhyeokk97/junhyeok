from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(10,10,1),  # (10, 10, 10)
                 strides=1,
                 padding='same',        # >> +0
                #  padding='valid',
                ))
model.add(Conv2D(filters=9, kernel_size=(3,3),      # (8, 8, 9)
                 strides=1,
                 padding='valid'))  # valid = defualt >> +1
model.add(Conv2D(8,4))  # 8 = filter, 4 = kernel_size >> 4 x 4   / (5, 5, 8)

model.summary()