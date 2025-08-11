from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np
import pandas as pd

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

path = './_data/kaggle/bike'
train_csv = pd.read_csv(path + '/train.csv')
test_csv =  pd.read_csv(path + '/test.csv')
submission_csv = pd.read_csv(path + '/sampleSubmission.csv')

def datetime_preprocessing(df):
    df = df.copy(deep=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['is_weekend'] = df['datetime'].dt.weekday.isin([5,6]).astype(int)
    df = df.drop(columns=['datetime'], axis=1)
    return df

train_csv = datetime_preprocessing(train_csv)
test_csv = datetime_preprocessing(test_csv)

x = train_csv.drop(columns=['casual','registered','count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=4245)

# 2. 모델 구성
model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=13))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 학습
early = EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True,
)

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.1, verbose=2, callbacks=[early])

# 4. 평가, 예측
results = model.predict(x_test)
rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

submission_csv['count'] = model.predict(test_csv)
submission_csv.to_csv(path+'/submission_early_stop.csv', index=False)

'''
validation 적용 후
RMSE : 80.44079201473821
R2 : 0.8023590353444101

EarlyStopping 적용 후
RMSE : 71.7445762977589
R2 : 0.8427819377362654
'''