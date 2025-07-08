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
                                                    test_size=0.1,
                                                    random_state=38)

model = Sequential()
model.add(Dense(128, input_dim=9))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=0,
          validation_split=0.2)

print("@@@@@")
print(hist)
print("@@@@@")
print(hist.history)
print('loss@@@@')
print(hist.history['loss'])
print('val_loss@@@@')
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))   # 9 x 6 사이즈 / 그래프 크기 설정.
plt.plot(hist.history['loss'], c='red', label='loss')  #  y값만 넣으면 시간 순으로 그림 / 이미지 출력하는법 다시 한 번 확인하고 암기하기
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('따릉이 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 우측 상단에 label 표시
plt.grid()  # 격자 표시
plt.show()

loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

r2 = r2_score(y_test, results)
def RMSE(y_test, y_predict):                                # def는 파이썬 용어로 함수 사용/ results 값이 y의 perdict값이다.
    return np.sqrt(mean_squared_error(y_test, y_predict))   # 재사용// sqrt로 루트 씌우고 RMSE에 돌려준다.

rmse = RMSE(y_test, results)    # y_test 값과 results 값 비교.

print('loss: ', loss)
print('r2 score: ', r2)
print('RMSE: ', rmse)

y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit



# epochs=147
# loss:  1746.6363525390625
# r2 score:  0.7180734810337477
# RMSE:  41.792778973159656

# validation 적용
# loss:  1780.4482421875
# r2 score:  0.7126158852639947
# # RMSE:  42.19535693273787
