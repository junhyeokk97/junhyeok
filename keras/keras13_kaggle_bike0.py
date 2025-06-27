from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터
path = './_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sampleSubmission_csv = pd.read_csv(path + 'sampleSubmission.csv')

#region(데이터 정보)
# print(train_csv.shape)              # (10886, 12)
# print(train_csv.columns)            
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],dtype='object')
# print(test_csv.shape)               # (6493, 9)
# print(test_csv.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed'], dtype='object')
# print(sampleSubmission_csv.shape)   #(6493, 2)
# print(train_csv.isna().sum())       # NaN 없음
#endregion

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)    # [10886 rows x 8 columns]
y = train_csv[['casual', 'registered', 'count']]
print(y)    # [10886 rows x 3 columns]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, shuffle=True, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
def RMSE(a, b) : return np.sqrt(mean_squared_error(a, b))
rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)

print('#############')
print('RMSE :', rmse)
print('R2 :', r2)
print('#############')

#제출 서류 만들기
y_submit = model.predict(test_csv)
sampleSubmission_csv['count'] = y_submit[:, 2]
sampleSubmission_csv.to_csv(path + 'kjh1.csv')




