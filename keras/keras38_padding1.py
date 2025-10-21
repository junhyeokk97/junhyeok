#[실습]

#100, 100, 3 이미지를 10, 10, 11 로 만들기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D

model = Sequential()
model.add(Conv2D(11, (2,2), input_shape=(100,100,3),
                 strides=1,
                 padding='same',
                ))
model.add(MaxPooling2D())
model.add(Conv2D(filters=(11), kernel_size=(3,3),
                 strides=1,
                 padding='valid'
                 ))
model.add(MaxPooling2D())
model.add(Conv2D(11,3))
model.add(MaxPooling2D())
model.add(Conv2D(11,2))

model.summary()