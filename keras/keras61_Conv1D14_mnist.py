# 36-6 copy

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten, Dropout, BatchNormalization, Reshape
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


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# #2. 모델구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape=(28, 28)))
model.add(Conv1D(31, 27))
model.add(Flatten())
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=14, activation='relu'))
model.add(Dropout(0.15))
model.add(BatchNormalization())
model.add(Dense(units=12))    # input_shape는 생략 되어 있었으므로 사용해도됨 .
model.add(Dense(units=10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=2, restore_best_weights=True)

# ################# mcp 세이브 파일명 만들기 ##################
# import datetime
# date = datetime.datetime.now()
# print(date)                     # 2025-06-10 12:39:16.850419
# print(type(date))               # <class 'datetime.datetime'>
# date = date.strftime('%m%d_%H%M')      # %m월 %d일, %H 시간, %M 분
# print(date)                     # 0610_1239
# print(type(date))               # <class 'str'>

# path = './_save/keras36_cnn5/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# filepath = "".join([path, 'k36_', date, '_', filename])

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto', verbose=1,
#     save_best_only=True, 
#     filepath=filepath
# )

str = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=250, verbose=2,
                 validation_split=0.2, callbacks=[es])
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # evaluate에도 verbose 사용 가능.
print('loss: ', loss[0])
print('acc: ', loss[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test.to_numpy(), axis=1)
acc = accuracy_score(y_test, y_predict)
print('걸린시간: ', end-str)


# loss:  0.06271646916866302
# acc:  0.9819999933242798
# (10000,) (10000,)
# accuracy_score:  0.982 


# MinMax
# loss:  0.04909356310963631
# acc:  0.9847000241279602
# (10000,) (10000,)
# accuracy_score:  0.9847

# 정규화
# loss:  0.053068749606609344
# acc:  0.9854000210762024
# (10000,) (10000,)
# accuracy_score:  0.9854

# 3
# loss:  0.05087979882955551
# acc:  0.9857000112533569
# (10000,) (10000,)
# accuracy_score:  0.9857

# lstm
# loss:  0.07693798840045929
# acc:  0.9818999767303467
# 걸린시간:  95.18015122413635

# conv1d
# loss:  0.18351492285728455
# acc:  0.9514999985694885
# 걸린시간:  49.45198369026184