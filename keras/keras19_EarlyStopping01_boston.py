#18-1 카피
"""
tf274 - keras  // anaconda 통설치

conda create -n tf274 python==3.9.7 andnaconda
tensorflow 2.7.4
numpy 1.20.3
scikit 0.24.2
matplotlib 3.4.3
pandas 1.3.4
"""
import sklearn as sk
from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets = load_boston()
print(datasets)
print(datasets.DESCR) #(506,13))
print(datasets.feature_names)
x = datasets.data
y = datasets.target

print(x)
print(x.shape) #(506, 13)
print(y)
print(y.shape) #(506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.1,
    random_state=814)
    
#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min',   # 최대값 max, 알아서 찾아줘 auto
                   patience = 200, restore_best_weights= True,)     # 최소 값을 작게 잡으면 최소지역에 대한 오류가 발생 할 수 있다.

hist = model.fit(x,y, epochs=100000, batch_size=32, verbose=0, validation_split=0.2, 
                 callbacks=[es])

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

print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict([x_test])

print("loss : ", loss)
print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))   # 9 x 6 사이즈 / 그래프 크기 설정.
plt.plot(hist.history['loss'], c='red', label='loss')  #  y값만 넣으면 시간 순으로 그림 / 이미지 출력하는법 다시 한 번 확인하고 암기하기
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('보스턴 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 우측 상단에 label 표시
plt.grid()  # 격자 표시
plt.show()


#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict([x_test])

print("loss : ", loss)
print("[x]의 예측값 : ", results)
r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)


# patience = 100  epochs=100000  restore_best_weights= True
# r2 스코어 :  0.42478765514770966

# patience = 200  epochs=100000  restore_best_weights= True
# r2 스코어 :  0.918118048845775