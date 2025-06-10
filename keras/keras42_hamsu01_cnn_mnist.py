import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Input
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

#  x reshape >> (60000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)   # 10000은 0번째 shape, 28은 각각 첫 번째 두 번째,1은 세 번째 shape
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])로 출력해도 됨.
print(x_train.shape, x_test.shape)   # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# #2. 모델구성
# model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(28, 28, 1)))   # 27,27,64
# model.add(Conv2D(filters=32, kernel_size=(3,3)))        # 25,25,32
# model.add(Conv2D(16, (3,3)))                            # 23,23,16
# model.add(Flatten())
# model.add(Dense(units=16))
# model.add(Dense(units=16))
# model.add(Dense(units=10, activation='softmax'))
# model.summary()

input = Input(shape=(28,28,1))
conv1 = Conv2D(64, (2,2))(input)
conv2 = Conv2D(32, (3,3))(conv1)
conv3 = Conv2D(16, (3,3))(conv2)
flat = Flatten()(conv3)
dense1 = Dense(16)(flat)
output = Dense(10, activation='softmax')(dense1)
model = Model(inputs=input, outputs=output)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=500, verbose=2,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
          verbose=2)

loss = model.evaluate(x_test, y_test, verbose=1)    # evaluate에도 verbose 사용 가능.
print('loss: ', loss[0])
print('acc: ', loss[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test.to_numpy(), axis=1)
acc = accuracy_score(y_test, y_predict)

print('accuracy_score: ', acc)


# loss:  0.07223323732614517
# acc:  0.9124000072479248
# accuracy_score:  0.9124