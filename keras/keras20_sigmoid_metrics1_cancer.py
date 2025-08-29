from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

#1. 데이터
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

#region(데이터 정보)
# print(dataset.DESCR)  # (569, 30)     # sklearn에서만 쓰는 명령어 .DESCR
# print(dataset.feature_names)
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
#  'mean smoothness' 'mean compactness' 'mean concavity'
#  'mean concave points' 'mean symmetry' 'mean fractal dimension'
#  'radius error' 'texture error' 'perimeter error' 'area error'
#  'smoothness error' 'compactness error' 'concavity error'
#  'concave points error' 'symmetry error' 'fractal dimension error'
#  'worst radius' 'worst texture' 'worst perimeter' 'worst area'
#  'worst smoothness' 'worst compactness' 'worst concavity'
#  'worst concave points' 'worst symmetry' 'worst fractal dimension']
# print(type(dataset))    # <class 'sklearn.utils.Bunch'>
#endregion

x = dataset.data    #(569, 30)
y = dataset.target  #(569,)

# M.L에는 1) 지도 학습과(supervised), 2) 비지도 학습(unsupervised)이 있다. (ps. 강화학습도 있음)
# 지도 학습에는 다시 1) 회귀 방식(r2 score)과, 2) 분류 방식(F1 score/Accuracy)이 있다.
# 분류 방식에는 다시 1) 이진 분류와, 2) 다중 분류가 있다.
# 비지도 학습은 현재 지도 학습 데이터 전처리에 주로 쓰인다.

#region(x, y 정보)
# print(x)
# print(y)
# print(type(x))  # <class 'numpy.ndarray'>     # 딕셔너리는 키밸류
# print(type(y))  # <class 'numpy.ndarray'>     #
# print(x.shape)  # (569, 30)
# print(y.shape)  # (569,)
#endregion

## y에서 0과 1의 개수가 몇 개인지 찾아보기.
#region
# 1. 넘파이로 찾을 때
# print(np.unique(y))    #[0 1]         # np.unique()
# print(np.unique(y, return_counts=True))     #(array([0, 1]), array([212, 357], dtype=int64))
# 2. 판다스로 찾을 때
# print(pd.value_counts(y))             # pd.value_counts()
# 1    357
# 0    212
# print(y.value_counts())               # y.value_counts() -> y는 numpy 여서 안됨
# print(pd.DataFrame(y).value_counts()) # pd.DataFrame()
# 1    357
# 0    212
# print(pd.Series(y).value_counts())    # pd.Series()
# 1    357
# 0    212
# 분류 문제에서 가장 중요한 것은 데이터의 갯수다. 불균형 데이터는 올바른 결과를 못 줄수도 있다.
# Accuracy 는 균형 데이터에서 사용하고, F1 score는 불균형 데이터에서 사용한다.
# 분류 문제에서는 항상 y 데이터 까서 확인해라.
#endregion

r = random.randint(1, 10000)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=r,
)

print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,   ) (171,   )

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=30, activation='relu'))  # activation='relu' output layer만 아니면 사용해라.
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))    
# model.add(Dense(1, activation='linear'))는 -무한대~무한대까지 나올 수 있음.
# 그래서 sigmoid 함수로 범위를 0<x<1로 한정한다.
# 이진 분류는 output layer에서 무조건 activation='sigmoid'
# 이진 분류에서 output layer의 노드수는 무조건 1개 

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')               # mse는 거리를 기준으로 하는데 sigmoid에는 의미없는 거리일 뿐임.
model.compile(loss='binary_crossentropy', optimizer='adam', # 이진 분류에는 100% binary_crossentropy를 쓴다.
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True
)

start_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=100000, batch_size=32,
    verbose=2, validation_split=0.2,
    callbacks=[es],
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# print(y_predict[:10])
y_predict = np.round(y_predict) # pyhon의 반올림
# print(y_predict[:10])

# print('[BCE, ACC] :', results)         # BCE : [0.13512155413627625, 0.9532163739204407] -> loss와 accuracy
# print('[BCE] :', results[0])                                        # [BCE] : 0.13985399901866913
# print('[ACC] :', results[1])                                        # [ACC] : 0.9415204524993896
print('[BCE](소숫점 4번째 자리까지 표시) :', round(results[0], 4))      # [BCE](소숫점 4번째 자리까지 표시) : 0.1399
print('[ACC](소숫점 4번째 자리까지 표시) :', round(results[1], 4))      # [ACC](소숫점 4번째 자리까지 표시) : 0.9415   

#region(그림)
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
# plt.title('암 loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.grid()
# plt.show()
#endregion

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
accuracy_score = np.round(accuracy_score, 4)
print("acc_score :", accuracy_score)
print('걸린 시간 :', round(end_time - start_time, 2), '초')








