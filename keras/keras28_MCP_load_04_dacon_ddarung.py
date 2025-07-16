import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.shape) # (1459, 10)
print(test_csv.shape) # (715, 9)
print(submission_csv.shape) # (715, 1)
print(train_csv.info())

train_csv = train_csv.fillna(train_csv.mean())
print(train_csv.isna().sum())
print(train_csv.info())

print(test_csv.info())   # test_csv 결측치는 절대 dropXXX
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=42)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# model = Sequential()
# model.add(Dense(25, input_dim=9, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(25, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(1, activation= 'linear'))

# # model.summary()

# # path = './_save/keras28_mcp/04_dacon_ddarung/'
# # model.save_weights(path + 'keras28_ddarung_save1.h5')

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

# path = './_save/keras28_mcp/04_dacon_ddarung/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # epoch 앞 4자리, val_loss 소수점 4자리까지
# filepath = "".join([path, 'k28_', date, '_', filename])

# print(filepath)

# # path = './_save/keras28_mcp/01_boston/'
# mcp = ModelCheckpoint(monitor= 'val_loss', mode= 'auto', save_best_only=True,
#                       filepath=filepath)

# model.fit(x_train,y_train, epochs=1000, batch_size=15, verbose=2, validation_split=0.2, 
#                  callbacks=[es, mcp])

path = './_save/keras28_mcp/04_dacon_ddarung/'
model = load_model(path + 'keras28_ddarung_save2.h5')



#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict(x_test)

print("loss : ", loss)
r2 = r2_score(y_test, results)
def RMSE(y_test, y_predict):                                # def는 파이썬 용어로 함수 사용/ results 값이 y의 perdict값이다.
    return np.sqrt(mean_squared_error(y_test, y_predict))   # 재사용// sqrt로 루트 씌우고 RMSE에 돌려준다.

rmse = RMSE(y_test, results)
print('r2 스코어 : ', r2)
print('RMSE: ', rmse)

# loss :  2389.020263671875
# r2 스코어 :  0.6935536226688566
# RMSE:  48.87760475727566

