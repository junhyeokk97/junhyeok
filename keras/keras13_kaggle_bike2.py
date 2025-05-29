#1. train_csv 에서 casual과 registered를 y로 잡는다.
#2. 훈련해서 test_csv의 casual과 regietered를 예측(predict한다.)
#3. 예측한 casual과 regitered를 test_csv 컬럼에 넣는다.
#   (N, 8)  >>  (N ,10)  test.csv 파일로 new_test.csv 파일을 만든다.


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

path = './_data/kaggle/bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['casual','registered', 'count'], axis= 1)

y = train_csv[['casual','registered']]
print(x)    # (10886, 8)
print(y)    
print(y.shape) # (10886, 2)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=55)

model = Sequential()
model.add(Dense(144, activation='relu', input_dim=8))
model.add(Dense(144, activation='relu'))
model.add(Dense(72, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=32, verbose=0)

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
test_csv[['casual','registered']] = y_submit
print(test_csv.shape)
print(y_submit)

# test_csv.to_csv( path + 'new_test.csv')
