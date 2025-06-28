import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

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
                                                    test_size=0.2,
                                                    random_state=42)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

model = Sequential()
model.add(Dense(25, input_dim=9, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(1, activation= 'linear'))

from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor= 'val_loss', mode= 'min',
                   patience = 25, restore_best_weights=True, min_delta=1e-4)

# hist = 
model.fit(x_train ,y_train , epochs=1000, batch_size=15,
                 validation_split=0.2, verbose=2, callbacks=[es])



# print("@@@@@")
# print(hist)
# print("@@@@@")
# print(hist.history)
# print('loss@@@@')
# print(hist.history['loss'])
# print('val_loss@@@@')
# print(hist.history['val_loss'])

# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm

loss = model.evaluate(x_test,y_test)
results = model.predict([x_test])

print("loss : ", loss)
r2 = r2_score(y_test, results)
def RMSE(y_test, y_predict):                                # def는 파이썬 용어로 함수 사용/ results 값이 y의 perdict값이다.
    return np.sqrt(mean_squared_error(y_test, y_predict))   # 재사용// sqrt로 루트 씌우고 RMSE에 돌려준다.

rmse = RMSE(y_test, results)
print('r2 스코어 : ', r2)
print('RMSE: ', rmse)

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))   # 9 x 6 사이즈 / 그래프 크기 설정.
# plt.plot(hist.history['loss'], c='red', label='loss')  #  y값만 넣으면 시간 순으로 그림 / 이미지 출력하는법 다시 한 번 확인하고 암기하기
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.title('따릉이 loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(loc='upper right')   # 우측 상단에 label 표시
# plt.grid()  # 격자 표시
# plt.show()

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit

submission_csv.to_csv( path + 'submission_0529_2.csv')

# patience = 100      epochs=10000
# r2 스코어 :  0.7133753476371598


# patience = 400      epochs=10000
# r2 스코어 :  0.8595960249011037
# RMSE:  29.493274129147547

# patience = 200        epochs=5000
# loss :  1203.54052734375
# r2 스코어 :  0.8057351736160792
# RMSE:  34.692082681328365


# patience = 200      epochs=2000
# loss :  1966.0738525390625
# r2 스코어 :  0.735584310334792

# patience = 25       epochs=1000     random_state=42