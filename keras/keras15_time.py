#14 카피

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import time # 시간에 대한 모듈 import


#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape) # (10,)
# print(y.shape) # (10,)

x_train = np.array(range(100))
y_train = np.array(range(100))

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 ,훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()    # 현재 시간을 반환, 시작시간.
# print(start_time)

model.fit(x_train, y_train, epochs=100, batch_size=32,
          verbose=1)
# verbose  = 0 => 로그 생략
#          = 1 => 디폴트
#          = 2 => 프로그레스 바 생략
#          = 3 => 에포만 뜸.

end_time = time.time()
print("걸린시간 : ", end_time - start_time, '초')

#1. 1000에포에서 0, 1, 2, 3의 시간을 적는다.

#2. 1000에포에서 verbose = 1의 시간을 적는다.
#batch 1, 32, 128의 시간
#batch = 1 / 47.04224896430969 초
#batch = 32 / 3.1004140377044678 초
#batch = 128 / 1.6348466873168945 초
