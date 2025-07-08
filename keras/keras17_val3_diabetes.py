import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x)
print(y)
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.1,
    random_state=813)

#2. 모델구성
model = Sequential()
model.add(Dense(40, input_dim=10))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

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

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))   # 9 x 6 사이즈 / 그래프 크기 설정.
plt.plot(hist.history['loss'], c='red', label='loss')  #  y값만 넣으면 시간 순으로 그림 / 이미지 출력하는법 다시 한 번 확인하고 암기하기
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('간암 loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')   # 우측 상단에 label 표시
plt.grid()  # 격자 표시
plt.show()

#4. 평가, 예측
loss= model.evaluate(x_test, y_test)
results = model.predict([x_test])

print('loss : ', loss)
r2 = r2_score(y_test, results)
print('r2 스코어 : ', r2)



# 기존
# loss :  2281.981201171875
# r2 스코어 :  0.6295709504734388 epchos=1000 

# validation 적용
# loss :  2384.9833984375
# r2 스코어 :  0.6128509322231721