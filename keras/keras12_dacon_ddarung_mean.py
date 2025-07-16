# https://dacon.io/competitions/open/235576/overview/description

import numpy as np                                           # 훈련에 특화됨.
import pandas as pd                                          # 데이터분석을 할 때 전처리 정제에서 유명함.
# print(np.__version__)                                        # 1.23.0
# print(pd.__version__)                                        # 2.2.3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/dacon/따릉이/'      # .(점 한개) = 현재 작업폴더 study25

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # a=b b를 a에 넣겠다. // index_col : 이 컬럼은 인덱스다
print(train_csv)                    # [1459 rows x 10 columns] = [1459,10]                  

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(test_csv)                     # [715 rows x 9 columns] = [715,9]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(submission_csv)               # [715 rows x 1 columns] = [715,9]  // Nan 데이터가 없음 : 결측치

print(train_csv.shape)              #(1459,10)
print(test_csv.shape)               #(715,9)
print(submission_csv.shape)         #(715,1)

print(train_csv.columns)
                                    # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                                    #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                                    #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                                    #       dtype='object')
print(train_csv.info())             # 결측치 확인가능
                                    # 총 1459개인데 컬럼마다 데이터가 빠져있다 데이터를 삭제하면 일부만 없어지더라도 그 행 가로줄이 다 삭제해야한다.
                                    # 최악의 경우 1459에서 하나도 안겹친 데이터가 다 빠져서 많은 데이터가 날라간다.
                                    # 빠진 데이터 위아래값하고 같게 하거나 평균값으로 데이터를 채워도 괜찮다
                                    # 하지만 실제 데이터하고 다를 수 있음 주의 요망 // 결측치 처리에 대한 생각을 많이 해야함.
                                    # 결측치 삭제 or 수정
print(train_csv.describe())

################################ 결측치 처리 2. 평균값 #####################################

# 결측치 자리 위치를 파악해서 특정값을 넣기
train_csv = train_csv.fillna(train_csv.mean())    # train_csv에 평균값을 결측치 자리에 넣겠다 (컬럼별 평균) / nan 자리에 컬럼 한 열의 평균값이 다 동일하게 들어간다.
print(train_csv.isna().sum())
print(train_csv.info())

################################ 테스트 데이터에 결측치 #####################################
print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

#################### 받은 데이터를 전처리 (결측치 정리) ######################

#################### x,y 데이터를 분리한다 ##########################

x = train_csv.drop(['count'], axis=1) # 축1번 count 컬럼을 삭제하고 남은 데이터를 x에 넣는다
                                      # count 라는 axis=1열 삭제, 참고로 행은 axis=0 (2차원)
                                      # train_csv에서 9개 컬럼만 복사해서 x에 넣어준거지 train_csv의 상태는 그대로다
                                      # x= count 제외한 데이터 y= count 데이터

print(x)                              # [1459 rows x 9 columns]                                

y = train_csv['count']               # count컬럼만 빼서 y에 넣겠다.

print(y.shape) # (1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.1,
    random_state=248
    )

#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=9))
model.add(Dense(80, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict(x_test)

print("loss : ", loss)
r2 = r2_score(y_test, results)
print('r2 : ', r2)

def RMSE(y_test, results):
    return np.sqrt(mean_squared_error(y_test, results)) 
rmse = RMSE(y_test, results) 
print('RMSE : ', rmse)

"""
목표
r2 0.58 이상
 loss 2400이하

 loss :  2149.123046875
 r2 스코어 : 0.6315198682324188

같은 파라미터로 돌려도 mean쪽이 더 잘나옴
"""

# submission.csv에 test_csv의 예측값 넣기
y_submit = model.predict(test_csv) # train 데이터의 shape와 동일한 컬럼을 확인하고 넣자
                        # x_train.shape: (N, 9)
# print(y_submit.shape) # (715, 1)

############## submission.csv 파일 만들기// count 컬럼값만 넣어주기 ####################
submission_csv['count'] = y_submit
# print(submission_csv)

######################## csv파일 만들기 ######################
# submission_csv.to_csv(path + 'submission_0522_1013.csv') # csv 만들기.
""""
loss :  1606.2760009765625  epochs=114
r2 :  0.7407292092618014
RMSE :  40.07837289822755


loss :  1600.2337646484375  epochs=115
r2 :  0.7417044826901299
RMSE :  40.00292244138169


loss :  1540.8961181640625  epochs=122
r2 :  0.7512822523564865
RMSE :  39.25424906322806
"""
