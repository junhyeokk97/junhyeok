import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import sklearn as sk

from sklearn.datasets import fetch_california_housing
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

dataset = fetch_california_housing()
print(dataset)
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x)
print(x.shape)      # (20640, 8)
print(y)
print(y.shape)      # (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=55)

model = Sequential()
model.add(Dense(15, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=100,
          validation_split=0.2, verbose=2)

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

loss = model.evaluate(x_test, y_test)
results = model.predict([x_test])

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, results)

r2= r2_score(y_test, results)
print('################################')
print('RMSE: ' ,rmse)
print('loss: ', loss)
print('r2_socre: ', r2)
print('[x]의 예측값: ', results)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('캘리포니아 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()


