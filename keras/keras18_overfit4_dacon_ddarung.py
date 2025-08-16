from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time

path ='./_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

# print(train_csv.isna().sum())
train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())
# print(train_csv.isna().sum())
# print(test_csv.isna().sum())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
print(x.shape)  #(1459, 9)
print(y.shape)  #(1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123
)

model = Sequential()
model.add(Dense(100, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
str_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=21,
          verbose=1, validation_split=0.2)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)
print('RMSE :', rmse)
print('R2 :', r2)
print('time :', end_time - str_time, '초')

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit
submission_csv.to_csv(path + '05_23_prac.csv', index=False)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('따릉이')
plt.xlabel('에포')
plt.ylabel('loss')
plt.grid()
plt.legend(loc='upper right')
plt.show()

# RMSE : 48.051277745433055
# R2 : 0.6129672443899667