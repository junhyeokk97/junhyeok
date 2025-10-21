from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# 원본은 N,5,5,1 이미지   / N은 데이터의 수              #  세로 , 가로 , 색깔
model = Sequential()                                    # height, width, channels
model.add(Conv2D(filters = 10,kernel_size =  (2,2), input_shape = (5,5,1))) # (2,2)는 커널사이즈 >> 2,2 사이즈로 잘라준다. # (4, 4, 10)   2 * 2 * 1 + 1 ) * 10 = 50
model.add(Conv2D(5, (2,2)))     # (3, 3, 5)         2 * 2 * 10 + 1 ) * 5 = 205       
model.add(Conv2D(10, (3,3)))    # (1, 1, 10)        3 * 3 * 5 + 1 ) * 10 = 460
model.add(Flatten())    # ( ,10)  /  연산량은 없음.
model.add(Dense(units=10))  # input = (batch_size, input_dim) / output = units
model.add(Dense(10))

model.summary()

