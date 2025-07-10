from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import sklearn as sk    #print(sk.__version__)   #1.6.1 -> 1.1.3
import numpy as np      #print(np.__version__)   #1.23.0

from sklearn.datasets import load_boston
#1. 데이터
dataset = load_boston()
# print(dataset)                  # load_boston() : 내용 보기
# print(dataset.DESCR)            # .DESCR :데이터 상세 확인
# print(dataset.feature_names)    #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

x = dataset.data
y = dataset.target

# print(x)
# print(x.shape)  #(506, 13)
# print(y)
# print(y.shape)  #(506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=243)

#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim = 13))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

def RMSE(y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)
print('###################')
print('RMSE :', rmse)   
print('R2 :', r2)       #->0.75 이상

# ###################
# RMSE : 4.827135674076012
# R2 : 0.756898103912625
# print(x_test.shape, y_test.shape)
# exit()

# 여기서는 이게 돌아갈리가 없음. x_test와 results의 열의 수가 다름
# import matplotlib.pyplot as plt
# plt.scatter(x_test, y_test)
# plt.plot(x_test, results, color='red')
# plt.show()