import sklearn as sk
print(sk.__version__) 

from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
datasets = load_boston()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=42)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(np.min(x_train), (np.max(x_train), )) # 0.0 (1.0000000000000002,)
print(np.min(x_test), (np.max(x_test), ))   # -0.06141956477526944 (1.0,)

model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(1))

model.summary()

# path = './_save/keras26/'
# # # model.save(path + 'keras26_1_save.h5')
# model.save_weights(path + 'keras26_5_save1.h5')

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min',   # 최대값 max, 알아서 찾아줘 auto
                   patience = 100, restore_best_weights= True,)     # 최소 값을 작게 잡으면 최소지역에 대한 오류가 발생 할 수 있다.

############# mcp 세이브 파일명 만들기 ##############
import datetime
date = datetime.datetime.now()
print(date)     # 2025-06-02 13:00:40.340100
print(type(date))   # <class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')
print(date) # 0602_1306
print(type(date))   # <class 'str'>   str = 문자열

path = './_save/keras27_mcp2/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # epoch 앞 4자리, val_loss 소수점 4자리까지
filepath = "".join([path, 'k27_', date, '_', filename])

print(filepath)
# ./_save/keras27_mcp2/k27_0602_1442_{epoch:04d}-{val_loss:.4f}.hdf5

mcp = ModelCheckpoint(monitor= 'val_loss', mode= 'auto', save_best_only=True,
                      filepath=filepath)

hist = model.fit(x,y, epochs=10000, batch_size=1, verbose=2, validation_split=0.2, 
                 callbacks=[es, mcp])



#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, results)

# print("[x]의 예측값 : ", results)
print("loss : ", loss)
print('rmse: ', rmse)
