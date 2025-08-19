from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time

path = './_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

#region
# print(train_csv.columns)
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
# print(train_csv)            #[10886 rows x 11 columns]
# print(train_csv.isna().sum())   #없음
#endregion

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv[['casual', 'registered']]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123
)

model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='linear'))

model.compile(loss='mse', optimizer='adam')
srt_time = time.time()
hist = model.fit(x_train, y_train,
          epochs=100, batch_size=32,
          verbose=2, validation_split=0.2)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('RMSE :', rmse)
print('R2 :', r2)
print('걸린 시간 :', end_time - srt_time, '초')
# print(test_csv) #[6493 rows x 9 columns]

y_submit = model.predict(test_csv)
test_csv_copy = test_csv.copy()
test_csv_copy[['casual', 'registered']] = y_submit
test_csv_copy.to_csv(path + 'new_test.csv', index=False)

# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.title('자전거 수요')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.grid()
# plt.show()

# RMSE : 95.5275391820992
# R2 : 0.39540151302317567