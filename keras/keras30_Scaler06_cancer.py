import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

print(type(datasets))   # <class 'sklearn.utils.Bunch'>

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

print(np.unique(y, return_counts=True)) # numpy로 찾았을때
# (array([0, 1]), array([212, 357], dtype=int64))

print(pd.value_counts(y))
# 1    357
# 0    212
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=42)

print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,) (171,)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = StandardScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(50, input_dim=30, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor = 'val_loss', mode = 'min',   # 최대값 max, 알아서 찾아줘 auto
                   patience = 15, restore_best_weights= True,)     # 최소 값을 작게 잡으면 최소지역에 대한 오류가 발생 할 수 있다.

# import datetime
# date = datetime.datetime.now()
# print(date)     # 2025-06-02 13:00:40.340100
# print(type(date))   # <class 'datetime.datetime'>
# date = date.strftime('%m%d_%H%M')
# print(date) # 0602_1306
# print(type(date))   # <class 'str'>   str = 문자열

# path = './_save/keras28_mcp/06_cancer/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # epoch 앞 4자리, val_loss 소수점 4자리까지
# filepath = "".join([path, 'k28_', date, '_', filename])

# print(filepath)

# # path = './_save/keras28_mcp/01_boston/'
# mcp = ModelCheckpoint(monitor= 'val_loss', mode= 'auto', save_best_only=True,
#                       filepath=filepath)

model.fit(x,y, epochs=1000, batch_size=32, verbose=2, validation_split=0.1, 
                 callbacks=[es])

# path = './_save/keras28_mcp/06_cancer/'
# model.save(path + 'keras28_cancer_save.h5')



#4. 평가, 예측
print("=======================================")
results = model.evaluate(x_test,y_test)

print('loss: ', round(results[0], 4)) # 0.0365
print('acc: ', round(results[1], 5)) # 0.98245
# [0.036508865654468536, 0.9824561476707458] >> [loss, accuracy]
# loss: 0.1256 - acc: 0.9429 - val_loss: 0.3427 - val_acc: 0.8333   4가지가 출력된다. [ val_loss, val_acc가 중요 ]

y_predict = model.predict(x_test)
print(y_predict[:10])
y_predict = np.round(y_predict)
print(y_predict)



# loss:  0.0619
# acc:  0.98246

# loss:  0.5546
# acc:  0.75439