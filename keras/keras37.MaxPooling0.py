import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# 스케일링 2. 정규화 ( 많이 씀 )
x_train = x_train/255.              # 0에 쏠릴 수 있는 단점.
x_test = x_test/255.
print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_test), np.min(x_test))   # 1.0 0.0

#  x reshape >> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)   # 10000은 0번째 shape, 28은 각각 첫 번째 두 번째,1은 세 번째 shape
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])로 출력해도 됨.
print(x_train.shape, x_test.shape)   # (60000, 28, 28, 1) (10000, 28, 28, 1)


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(Conv2D(64, (3,3), strides=1, input_shape=(28, 28, 1)))   # 26,26,64
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(3,3)))                   # 24,24,64
model.add(MaxPooling2D())
model.add(Dropout(0.2))     # Conv layer도 Dropout 사용 가능.
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu'))                    # 22,22,32
model.add(Flatten())
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(units=16, input_shape=(16,)))    # input_shape는 생략 되어 있었으므로 사용해도됨 .
model.add(Dense(units=10, activation='softmax'))
model.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 64)        640
#  conv2d_1 (Conv2D)           (None, 24, 24, 64)        36928
#  conv2d_2 (Conv2D)           (None, 22, 22, 32)        18464
#  flatten (Flatten)           (None, 15488)             0
#  dense (Dense)               (None, 16)                247824
#  dense_1 (Dense)             (None, 16)                272
#  dense_2 (Dense)             (None, 10)                170
# =================================================================
# Total params: 304,298
# Trainable params: 304,298
# Non-trainable params: 0

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=2, restore_best_weights=True)

################# mcp 세이브 파일명 만들기 ##################
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')      # %m월 %d일, %H 시간, %M 분


path = './_save/keras37/'
filename = '.hdf5'
filepath = "".join([path, 'k37_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, 
    filepath=filepath)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=250, verbose=2,
                 validation_split=0.2, callbacks=[es, mcp])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # evaluate에도 verbose 사용 가능.
print('loss: ', loss[0])
print('acc: ', loss[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test.to_numpy(), axis=1)
print(y_predict.shape, y_test.shape)
acc = accuracy_score(y_test, y_predict)

print('accuracy_score: ', acc)

# 함수형
# loss:  0.04729847237467766
# acc:  0.9866999983787537
# accuracy_score:  0.9867

# MaxPooling
# loss:  0.03621238097548485
# acc:  0.9891999959945679
# accuracy_score:  0.9892

# loss:  0.027844039723277092
# acc:  0.9915000200271606
# accuracy_score:  0.9915