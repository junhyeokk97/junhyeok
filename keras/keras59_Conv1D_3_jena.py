import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape, Dropout, MaxPooling1D, BatchNormalization, SimpleRNN, GRU, Conv1D, Flatten
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
# RMSE
path = './_data/kaggle/jena/'
csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
test_csv = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')


csv = csv.replace(-9999, np.nan)
csv = csv.fillna(csv.median())
train_csv = csv.to_numpy()
# print(train_csv.isna().sum())
# print(train_csv.info())

timesteps = 288


def split(train_csv, timesteps, stride=1):
    aa = []
    for i in range(0, len(train_csv) - timesteps + 1, stride):
        subset = train_csv[i : (i+timesteps)].astype(np.float32)
        aa.append(subset)
        
    return np.array(aa)

aa = split(train_csv, timesteps=timesteps, stride=10)
# # print(aa.shape)



# # print(aa.shape)    # (420408, 144, 14)

x = aa[:-144, :-144, :-1]
y = aa[:-144, 144:, -1]
test = aa[-1, :-144 ,:-1].reshape(-1, 144, 13)
# # print(x)
# # print(y)
print(x.shape)  # (420120, 144, 13)
print(y.shape)  # (420120, 144)
print(test.shape)  # (1, 144, 13)

np_path = 'c:/study25/_data/_save_npy/'
# np.save(np_path + "keras56_01_x.npy", arr=x)
# np.save(np_path + "keras56_01_y.npy", arr=y)
# np.save(np_path + "keras56_01_test.npy", arr=test)

x = np.load(np_path + "keras56_01_x.npy")
y = np.load(np_path + "keras56_01_y.npy")
test = np.load(np_path + "keras56_01_test.npy")

# print(x.shape, y.shape)
# print(test.shape)


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=50,
                                                    test_size=0.2,
                                                    shuffle=True)



rbs = RobustScaler()
y_train = rbs.fit_transform(y_train)
y_test = rbs.fit_transform(y_test)

model = Sequential()
model.add(Conv1D(45, kernel_size=2, input_shape=(144,13)))
model.add(Conv1D(30, 2))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(7, activation='relu'))
model.add(Dense(144))
model.summary()

from tensorflow.keras.losses import Huber
huber = Huber(delta=1.0)
model.compile(loss=huber, optimizer='adam', metrics=['mse'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=24,
                   verbose=2, restore_best_weights=True)

import datetime
date = datetime.datetime.now()
# print(date)    
# print(type(date))   # <class 'datetime.datetime'>
date = date.strftime('%m%d_%H%M')
# print(date)
# print(type(date))   # <class 'str'>   str = 문자열

path = './study25/_data/kaggle/jena/weight/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # epoch 앞 4자리, val_loss 소수점 4자리까지
filepath = "".join([path, 'zz_', date, '_', filename])

model.fit(x_train, y_train, batch_size=150, epochs=1000,validation_split=0.2,
          verbose=2, callbacks=es)

path = './study25/_data/kaggle/jena/model/'
model.save(path + 'model_2.h5')

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_submit = model.predict(test)

y_predict = rbs.inverse_transform(y_predict)
y_submit = rbs.inverse_transform(y_submit)

def calc_wd_deg(u, v):
    wd = np.arctan2(-u, -v) * 180 / np.pi
    wd = (wd + 360) % 360
    return wd

print(y_submit.shape)  # 반드시 (1, 144) 출력되어야 함

if y_submit.ndim > 1:
    y_submit = y_submit.flatten()
    
print(y_submit.shape)  # (144,) 되어야 함

# 제출파일 행 길이 맞추기
sub_csv = sub_csv.iloc[:len(y_submit)].copy()

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
r2 = r2_score(y_test, y_predict)
rmse = RMSE(y_test, y_predict)
print('loss: ', loss)
print('RMSE: ', rmse)

path = './study25/_data/kaggle/jena/'
sub_csv['wd (deg)'] = y_submit
try:
    sub_csv.to_csv(path + 'jena_최준혁5_submit.csv', index=False)
    print("파일 저장 성공")
except Exception as e:
    print("파일 저장 실패:", e)
