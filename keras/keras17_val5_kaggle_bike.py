# https://www.kaggle.com/competitions/bike-sharing-demand/submissions

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/kaggle/bike/'
# path = '.\_data\kaggle\bike  >> \ 사용 시 주의 \n, \a, \b 등 예약된 단어들 제외하고 다 가능.
# path = '.\\_data\\kaggle\\bike\\'

# path = 'c:/Study25/_data/kaggle/bike/' >> 절대 경로

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'samplesubmission.csv')
# print(submission_csv)       # (N, )


print(train_csv)
print(train_csv.shape)  # (10886, 11)
print(test_csv.shape)   # (6493, 8)
print(submission_csv.shape) # (6493, 1)

print(train_csv.columns)    # Index(['season', 'holiday', 'workingday', 'weather', 'temp',
                            # 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],

print(test_csv.columns)     # Index(['season', 'holiday', 'workingday', 'weather', 'temp',
                            # 'atemp', 'humidity', 'windspeed']
                            
print(submission_csv.columns)   # Index(['count']

print(train_csv.info()) #  #   Column      Non-Null Count  Dtype
                        # ---  ------      --------------  -----
                        #  0   datetime    10886 non-null  object
                        #  1   season      10886 non-null  int64
                        #  2   holiday     10886 non-null  int64
                        #  3   workingday  10886 non-null  int64
                        #  4   weather     10886 non-null  int64
                        #  5   temp        10886 non-null  float64
                        #  6   atemp       10886 non-null  float64
                        #  7   humidity    10886 non-null  int64
                        #  8   windspeed   10886 non-null  float64
                        #  9   casual      10886 non-null  int64
                        #  10  registered  10886 non-null  int64
                        #  11  count       10886 non-null  int64

print(train_csv.isnull().sum()) # None 결측치 없음
                                # datetime      0
                                # season        0
                                # holiday       0
                                # workingday    0
                                # weather       0
                                # temp          0
                                # atemp         0
                                # humidity      0
                                # windspeed     0
                                # casual        0
                                # registered    0
                                # count         0
                                
print(test_csv.isna().sum())    # 결측치 없음.

print(train_csv.describe())     # 이상치 확인
#              season       holiday    workingday       weather         temp         atemp      humidity     windspeed        casual    registered         count
# count  10886.000000  10886.000000  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000  10886.000000
# mean       2.506614      0.028569      0.680875      1.418427     20.23086     23.655084     61.886460     12.799395     36.021955    155.552177    191.574132
# std        1.116174      0.166599      0.466159      0.633839      7.79159      8.474601     19.245033      8.164537     49.960477    151.039033    181.144454
# min        1.000000      0.000000      0.000000      1.000000      0.82000      0.760000      0.000000      0.000000      0.000000      0.000000      1.000000
# 25%        2.000000      0.000000      0.000000      1.000000     13.94000     16.665000     47.000000      7.001500      4.000000     36.000000     42.000000
# 50%        3.000000      0.000000      1.000000      1.000000     20.50000     24.240000     62.000000     12.998000     17.000000    118.000000    145.000000
# 75%        4.000000      0.000000      1.000000      2.000000     26.24000     31.060000     77.000000     16.997900     49.000000    222.000000    284.000000
# max        4.000000      1.000000      1.000000      4.000000     41.00000     45.455000    100.000000     56.996900    367.000000    886.000000    977.000000

### x, y 분리 ###

x = train_csv.drop(['casual','registered', 'count'], axis= 1) # 0=행 1=열
print(x)    # [10886 rows x 8 columns] >> (10886, 8)
y = train_csv['count']
print(y)
print(y.shape)    # (10886,)                pandas의 데이터 형태는 시리즈(1차원)와 데이터프레임(2차원) 2가지로 나뉜다.

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=38)

model = Sequential()
model.add(Dense(120, activation='relu', input_dim= 8))
model.add(Dense(148, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(52, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))     # linear = x=wa+b, 생략해도 됨

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2)

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
plt.title('바이크 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 우측 상단에 label 표시
plt.grid()  # 격자 표시
plt.show()

exit()

loss = model.evaluate(x_test, y_test)



results = model.predict([x_test])
r2 = r2_score(y_test, results)

def RMSE(y_test,results):
    return np.sqrt(mean_squared_error(y_test, results))
rmse = RMSE(y_test, results)

print('loss: ', loss)
print('r2 : ', r2)
print('RMSE: ', rmse)

y_submit = model.predict(test_csv)
# print(y_submit.csv)   # (N, )

submission_csv['count'] = y_submit 

# loss:  21312.07421875
# r2 :  0.34366044370724225
# RMSE:  145.9865661944495     model.add(Dense(180, activation='relu'))
#                              model.add(Dense(120, activation='relu'))
#                              model.add(Dense(100, activation='relu'))
#                              model.add(Dense(50, activation='relu'))              


# validation 적용


