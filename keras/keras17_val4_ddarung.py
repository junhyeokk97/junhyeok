 #https://dacon.io/competitions/open/235576/overview/description

import numpy as np   
import pandas as pd 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 데이터 (전처리 수업)
path = './_data/dacon/따릉이/'         
train_csv = pd.read_csv(path +'/train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col = 0)

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

print(y.shape)

x_train, x_test, y_train, y_test =  train_test_split (x, y,
                                                      test_size = 0.1,
                                                      random_state=0
                                                        )

#모델 생성 
model = Sequential()
model.add(Dense(2048, input_dim = 9))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#학습
model.compile(loss = 'mse', optimizer = 'adam') #(v)
model.fit(x_test, y_test, epochs=300, batch_size=4, validation_split=0.1) #(v)

#출력
loss = model.evaluate(x_test, y_test) #(v)
results = model.predict(x_test)#(v)

r2 = r2_score(y_test, results)#(v)
def RMSE (y_test, results):#(v)
    return np.sqrt(mean_squared_error(y_test, results))  #(v)

rmse = RMSE(y_test, results)
print('RMSE:', rmse)
print("loss출력값:", loss)
print("r2스코어:", r2)

# RMSE: 43.146914212601075
# loss출력값: 1861.656005859375
# r2스코어: 0.6995080021165584


# RMSE: 42.989577145363405
# loss출력값: 1848.1036376953125
# r2스코어: 0.7016955202251771

# RMSE: 43.27245648699534
# loss출력값: 1872.505615234375
# r2스코어: 0.6977568070682394

# RMSE: 44.021637710163816
# loss출력값: 1937.90478515625
# r2스코어: 0.6872006662337102


# RMSE: 49.57221267451186
# loss출력값: 2457.404296875
# r2스코어: 0.6033476448949411


# RMSE: 53.78141776571214
# loss출력값: 2892.44091796875
# r2스코어: 0.543644978606007

# RMSE: 51.205186314645196
# loss출력값: 2621.97119140625
# r2스코어: 0.6283532787390061

# validation 적용 후
# RMSE: 47.22288712924896
# loss출력값: 2230.0009765625
# r2스코어: 0.6759889208648263