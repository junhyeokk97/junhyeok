import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

print(type(datasets))   # <class 'sklearn.utils.Bunch'>

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (569, 30) (569,)
print(type(x))  # <class 'numpy.ndarray'>

print(x)
print(y)

# 0과 1의 갯수가 몇 개인지 찾아보기.

print(np.unique(y, return_counts=True)) # numpy로 찾았을때
# (array([0, 1]), array([212, 357], dtype=int64))

print(pd.value_counts(y))
# 1    357
# 0    212
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                     test_size=0.3,
                                                     random_state=7275,
                                                     shuffle=True,)
print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,) (171,)

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=30, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))        # 0,1의 중간인 0.5를 기준으로 잡는다.
                                                 # 아웃풋 레이어에는 relu 사용하지 않고
                                                 # 0.5미만 음수 무한대까지는 0으로 취급 0.5이상 양수 무한대까지는 1로 취급한다.
                                                 # sigmoid는 0과 1에 완벽히 수렴 할 수는 없다. / 이진분류는 simoid 사용.

model.compile(loss='binary_crossentropy', optimizer='adam',     # sigmoid를 사용할땐 mse를 사용 할 수 없음. / binary_crossentropy 사용
              metrics=['acc'])    # accuracy   /    metrics=['acc']는 훈련에 영향을 미치진 않는다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'val_loss', mode = 'min',   # 최대값 max, 알아서 찾아줘 auto
                   patience = 10, restore_best_weights= True,)     # 최소 값을 작게 잡으면 최소지역에 대한 오류가 발생 할 수 있다.

start_time = time.time()
hist = model.fit(x,y, epochs=100, batch_size=32, verbose=2, validation_split=0.2, 
                 callbacks=[es])

end_time = time.time()

#4. 평가, 예측
print("=======================================")
results = model.evaluate(x_test, y_test)
print(results)
print('loss: ', round(results[0], 4)) # 0.0365
print('acc: ', round(results[1], 5)) # 0.98245
# [0.036508865654468536, 0.9824561476707458] >> [loss, accuracy]
# loss: 0.1256 - acc: 0.9429 - val_loss: 0.3427 - val_acc: 0.8333   4가지가 출력된다. [ val_loss, val_acc가 중요 ]

y_predict = model.predict(x_test)
print(y_predict[:10])
y_predict = np.round(y_predict)
print(y_predict)









exit()
from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
print('acc_score: ', accuracy_score)
print('걸린시간: ', round(end_time-start_time, 2), '초')












