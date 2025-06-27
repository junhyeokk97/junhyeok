import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=42)


# model = Sequential()
# model.add(Dense(14, input_dim=10, activation='relu'))
# model.add(Dense(36, activation='relu'))
# model.add(Dense(18, activation='relu'))
# model.add(Dense(6, activation='relu'))
# model.add(Dense(1))

# # model.summary()

# path = './_save/keras28_mcp/03_diabetes/'
# model.save_weights(path + 'keras28_diabetes_save1.h5')

# #3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer = 'adam')

# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor = 'val_loss', mode = 'min',   # 최대값 max, 알아서 찾아줘 auto
#                    patience = 15, restore_best_weights= True,)     # 최소 값을 작게 잡으면 최소지역에 대한 오류가 발생 할 수 있다.

# import datetime
# date = datetime.datetime.now()
# print(date)     # 2025-06-02 13:00:40.340100
# print(type(date))   # <class 'datetime.datetime'>
# date = date.strftime('%m%d_%H%M')
# print(date) # 0602_1306
# print(type(date))   # <class 'str'>   str = 문자열

# path = './_save/keras28_mcp/03_diabetes/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # epoch 앞 4자리, val_loss 소수점 4자리까지
# filepath = "".join([path, 'k28_', date, '_', filename])


# # path = './_save/keras28_mcp/01_boston/'
# mcp = ModelCheckpoint(monitor= 'val_loss', mode= 'auto', save_best_only=True,
#                       filepath=filepath)

# hist = model.fit(x,y, epochs=90, batch_size=1, verbose=2, validation_split=0.1, 
#                  callbacks=[es, mcp])

path = './_save/keras28_mcp/03_diabetes/'
model = load_model(path + 'keras28_diabetes_save2.h5')



#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, results)

print("loss : ", loss)
print('rmse: ', rmse)

# loss :  2619.51318359375
# rmse:  51.18118151458486
