import numpy as np
import pandas as pd
print(np.__version__)   # 1.23.0
print(pd.__version__)   # 2.2.3


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터

path = './_data/dacon/따릉이/'     # 문자+문자 = 문자를 이어서 사용 ( ab + cd = abcd )
train_csv = pd.read_csv(path + 'train.csv', index_col=0)  #index_col=0 >> 1번째 열은 인덱스다.
print(train_csv)        # (1459, 11) [1459 rows x 11 columns] 
                        #  >> (1459, 10) [1459 rows x 10 columns] ID는 index이기 때문에 포함하지 않는다   # X와 y 분리
                                                                                                        #   X = train_csv.drop('count', axis=1)
                                                                                                         #   y = train_csv['count']

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # >> count 값
print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
print(submission_csv)   # [715 rows x 1 columns]  >> 측정 완료 된 값.

print(train_csv.shape) # (1459, 10)
print(test_csv.shape) # (715, 9)
print(submission_csv.shape) # (715, 1)

print(train_csv.columns) # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                         # 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                         # 'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                         # dtype='object')
print(train_csv.info())

# print(train_csv.describe())

# 1,2는 train
############## 결측치 처리 1. 삭제 ################
# print(train_csv.isnull().sum())   # 결측치의 개수 출력
# print(train_csv.isna().sum())

# train_csv = train_csv.dropna()  # train_csv에 있는 결측치를 삭제하고 다시 그 값을 옮긴다.
# print(train_csv.isna().sum())
# print(train_csv.info())
# print(train_csv)        # [1328 rows x 10 columns] >> 결측치를 뺀 값


############## 결측치 처리 2. 평균값 넣기 ################
train_csv = train_csv.fillna(train_csv.mean())
print(train_csv.isna().sum())
print(train_csv.info())


############## test ################
print(test_csv.info())   # test_csv 결측치는 절대 dropXXX
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())


x = train_csv.drop(['count'], axis=1)   # 행 또는 열 삭제  /  count라는 axis=1 삭제, 행은 axis=0, 열은 axis=1
print(x)    # [1459 rows x 9 columns] >> 결측치를 뺀다면 [1328 x 9]

y = train_csv['count']   # count 컬럼만 빼서 y에 넣겠다.
print(y.shape)      # (1459,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.25,
                                                    random_state=19)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=9))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor= 'val_loss', mode='min', patience=200,
                   restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10000,
                 batch_size=300, verbose=2, validation_split=0.2,
                 callbacks=[es])

loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

r2 = r2_score(y_test, results)
def RMSE(y_test, y_predict):                                # def는 파이썬 용어로 함수 사용/ results 값이 y의 perdict값이다.
    return np.sqrt(mean_squared_error(y_test, y_predict))   # 재사용// sqrt로 루트 씌우고 RMSE에 돌려준다.

rmse = RMSE(y_test, results)    # y_test 값과 results 값 비교.

print('loss: ', loss)
print('r2 score: ', r2)
print('RMSE: ', rmse)


# RMSE:  43.6679202141922
# loss:  1906.88720703125
# r2 score:  0.6922072080894514
"""
# test_size=0.1   random_state=25
# loss:  2831.338134765625
# r2 score:  0.5692420000723689


# test_size=0.1   random_state=38   batch_size=10
# loss:  2165.1982421875
# r2 score:  0.6505129421366024

# loss:  1947.177001953125
# r2 score:  0.685704016225781

# RMSE:  43.962022157603535
# loss:  1932.659423828125
# r2 score:  0.6880472989090888
"""

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv)  
                        # train 데이터의 shape와 동일한 컬럼을 확인하고 넣는다.
                        # x_train.shape:(N, 9),       N = Nan
                        
# print(y_submit.shape)   # (715, 1)

########### submission.csv 파일 만들기 // count컬럼 값만 넣어주기
# print(submission_csv)
submission_csv['count'] = y_submit
# print(submission_csv)

### csv 파일 만들기
# submission_csv.to_csv(path + 'submission_0523_1639.csv')    # csv 만들기










"""
random_state=233
loss:  2417.681640625
r2 score:  0.6064878031006418
RMSE:  49.1699255621158


random_state=12
loss:  2389.74267578125
r2 score:  0.5826826942527077
RMSE:  48.884995305818954


loss:  2400.9072265625
r2 score:  0.5645777988407203
RMSE:  48.999051708308606
"""


# relu 사용 후
# loss:  1809.8914794921875
# r2 score:  0.7078634118748528
# RMSE:  42.542818804487865


# epochs=145
# loss:  1753.3914794921875
# r2 score:  0.7169831411217651
# RMSE:  41.87351696414825


# epochs=147
# loss:  1746.6363525390625
# r2 score:  0.7180734810337477
# RMSE:  41.792778973159656