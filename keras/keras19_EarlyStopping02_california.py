import sklearn as sk
from sklearn.datasets import fetch_california_housing
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

datasets = fetch_california_housing()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets.data
y = datasets.target

print(x)
print(x.shape)  # (20640, 8)
print(y)
print(y.shape)  # (20640,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=814)

model = Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min',
                   patience = 100, restore_best_weights=True)
                   
hist = model.fit(x,y, epochs=100000, batch_size=32, verbose=0,
          validation_split=0.2, callbacks=[es])

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

print('loss: ', loss)
r2 = r2_score(y_test, results)
print('r2_score: ', r2)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))   # 9 x 6 사이즈 / 그래프 크기 설정.
plt.plot(hist.history['loss'], c='red', label='loss')  #  y값만 넣으면 시간 순으로 그림 / 이미지 출력하는법 다시 한 번 확인하고 암기하기
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('캘리포니아 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 우측 상단에 label 표시
plt.grid()  # 격자 표시
plt.show()