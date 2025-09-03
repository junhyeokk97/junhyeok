
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
datasets = load_iris()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) #(150, 4) (150,)
# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
# print(np.unique(y, return_counts=True))
# print(pd.DataFrame(y).value_counts())
# print(pd.Series(y).value_counts())
# print(pd.value_counts(y))
# 0    50
# 1    50
# 2    50

######OneHotEncoding(반드시 다중분류, y만)######
#1. sklearn용
from sklearn.preprocessing import OneHotEncoder     # 다중분류 : OneHotEncoder 무조건

# print(y.shape)            # (150,)
                            # 데이터의 값과 순서는 바뀌지 말아야 한다. 

y = y.reshape(-1,1)
OHE=OneHotEncoder(sparse=False)
OHE.fit(y)
y = OHE.transform(y)

# OHE = OneHotEncoder()
# y = y.reshape(-1,1)         # 매트릭스 형태를 받기 때문에 (n,1)로 reshape 해야 한다. (-1, 1) = (n, 1)
# y = OHE.fit_transform(y)
# # print(y)                  # 희소행렬 방식
# # print(y.shape)            # (150, 3)
# # print(type(y))            # <class 'scipy.sparse.csr.csr_matrix'> ->
# y = y.toarray()             # OHE.fit_transfor(y)의 데이터가 scipy 데이터 형식이기 때문에 pandas나 numpy 형식으로 바꿔주어야 한다.
# # print(type(y))            # <class 'numpy.ndarray'>

#2. pd용
# y = pd.get_dummies(y)
# print(y)
# print(y.shape)  #[150 rows x 3 columns]

#3. keras용
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)  #(150, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=55
)

model = Sequential()
model.add(Dense(100, input_dim=4, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))                                           # 다중분류 : output layer 노드의 개수는 y의 라벨 개수와 같다.
                                                                                    # 다중분류 : activation='softmax' 무조건
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])   # 다중분류 : loss='categorical_crossentropy' 무조건 
ES = EarlyStopping(
    monitor='val_loss', mode='min', patience=10,
    restore_best_weights=True
)
str_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=32,
                 verbose=2, validation_split=0.1,
                 callbacks=[ES])
end_time = time.time()

loss = model.evaluate(x_test, y_test)
print('loss :', loss[0])
print('Acc  :', loss[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_predict.shape, y_test.shape)
acc = accuracy_score(y_test, y_predict)

print('accuracy_score: ', acc)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('꽃')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# loss : 0.0022166292183101177
# Acc  : 1.0
# RMSE : 0.04708109194050322
# time : 10.9 sec