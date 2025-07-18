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


### x, y 분리 ###

x = train_csv.drop(['casual','registered', 'count'], axis= 1) # 0=행 1=열
print(x)    # [10886 rows x 8 columns] >> (10886, 8)
y = train_csv['count']
print(y)
print(y.shape)    # (10886,)                pandas의 데이터 형태는 시리즈(1차원)와 데이터프레임(2차원) 2가지로 나뉜다.

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.25,
                                                    random_state=37)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim= 8))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))     # linear = x=wa+b, 생략해도 됨

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor= 'val_loss', mode='min', patience=100,
                   restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000,
                 batch_size=100, verbose=2, validation_split=0.3,
                 callbacks=[es])

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

submission_csv['count'] = y_submit # y_submit에 submission_csv에 있는 count 값을 넣어라
# print(submission_csv)

# submission_csv.to_csv( path + 'submission_0523_18.csv', index=False)      # 첫번째 행에서 인덱스를 지우기 위해서는 index=False 를 사용한다.





# loss:  21805.6953125          random_state=55
# r2 :  0.3284585286185687      epochs=100, batch_size=32
# RMSE:  147.66753417460737


# loss:  21578.53515625         epochs=155
# r2 :  0.3354544994649319
# RMSE:  146.89633648306378

# loss:  21520.91796875
# r2 :  0.3372286569715385      model.add(Dense(90, activation='relu'))
# RMSE:  146.700118605662       model.add(Dense(1, activation='linear')

# loss:  22093.091796875
# r2 :  0.31960795538615805
# RMSE:  148.63744021769995






# loss:  21417.314453125
# r2 :  0.3404195541747147
# RMSE:  146.3465503776223

# loss:  21038.26171875
# r2 :  0.35209316799080725
# RMSE:  145.0457084695893

                            # model.add(Dense(80, activation='relu'))
                            
                            
                            
# loss:  21312.07421875
# r2 :  0.34366044370724225
# RMSE:  145.9865661944495     model.add(Dense(180, activation='relu'))
#                              model.add(Dense(120, activation='relu'))
#                              model.add(Dense(100, activation='relu'))
#                              model.add(Dense(50, activation='relu'))              