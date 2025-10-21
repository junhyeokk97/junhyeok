import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
train_datagen = ImageDataGenerator(
    rescale=1./255, # 0~255 스케일링, 정규화
    # horizontal_flip=True,    # 수평 반전 > 데이터 증폭 또는 변환.
    # vertical_flip=True,      # 수직 반전 > 데이터 증폭 또는 변환
    # width_shift_range=0.1,   # 평행 이동 10% > 데이터 증폭 또는 변환
    # height_shift_range=0.1,  # 수직 이동
    # rotation_range=5,        # 5도 회전
    # zoom_range=1.2,          # 1.2배 확대
    # shear_range=0.7,         # 좌표 하나를 고정 시키고, 다른 몇 개의 좌표를 이동 시키는 변환
    #                          # 이미지를 늘려서 증폭 또는 변환.
    # fill_mode='nearest'
)    # rescale 아래 부분들은 증폭이기 때문에 이미지가 너무 크게 보인다면 리스케일만 사용

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

train_path = './_data/image/brain/train/'
test_path = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    train_path,                 # 경로
    target_size=(200, 200),     # 리사이즈, 사이즈 규격 일치, 데이터가 크면 축소/ 작으면 확대
    batch_size=8,              # 이미지 파일을 배치 사이즈로 묶어서 훈련. / 160개의 이미지에 사이즈 10으로 진행 시, 16번의 훈련.
    class_mode='binary',        # 
    color_mode='grayscale',     # 
    shuffle=True,
)
    # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    test_path,                 # 경로
    target_size=(200, 200),     # 리사이즈, 사이즈 규격 일치, 데이터가 크면 축소/ 작으면 확대
    batch_size=8,              # 이미지 파일을 배치 사이즈로 묶어서 훈련. / 160개의 이미지에 사이즈 10으로 진행 시, 16번의 훈련.
    class_mode='binary',        # 
    color_mode='grayscale',     # 
    # shuffle=True,             # 평가는 셔플X / 셔플의 디폴트는 False
)
    # Found 120 images belonging to 2 classes.
    
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) # (16, 200, 200, 1) (16,)
print(x_test.shape, y_test.shape)   # (12, 200, 200, 1) (12,)

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

model = Sequential()
model.add(Conv2D(64, (3,3), strides=1, input_shape=(200,200,1)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=10))
model.add(Dense(units=1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', patience=30,
                   restore_best_weights=True)


# path = './_save/keras28_mcp/02_california/'
# filename = '02_california_{epoch:04d}_{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     filepath=path+filename
# )

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])


start = time.time()
model.fit(x_train, y_train, epochs=100000, validation_split=0.2,
          verbose=2, callbacks=[es])
end = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
# f1 = f1_score(y_test, y_predict)

print('loss: ', loss[0])
print('acc: ', loss[1])
# print('f1: ', f1)
print('time: ', end-start)

import matplotlib.pyplot as plt
plt.imshow(x_train[1], 'gray')
plt.show()

# loss:  0.3523506820201874
# acc:  0.8125
# time:  9.08041524887085


# loss:  0.345955491065979
# acc:  1.0
# time:  5.6581597328186035