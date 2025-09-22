# 27-3 카피

import sklearn as sk
print(sk.__version__)       #0.24.2

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

#1. 데이터
from sklearn.datasets import load_boston
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True,
    random_state=333,
)

mms = MinMaxScaler()
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)

# #2. 모델 구성 (load_weight에는 모델이 있어야 함)
# model = Sequential()
# model.add(Dense(10, input_dim=13))
# model.add(Dense(11))
# model.add(Dense(12))
# model.add(Dense(13))
# model.add(Dense(1))

# 위에 모델 구성 대신에 아래 load_model()을 씀
from tensorflow.python.keras.models import load_model
path = './_save/keras27_mcp/'
model = load_model(path + 'keras27_mcp3.hdf5')              # #2와 #3 가 저장 됨
# model.load_weights(path + 'keras26_5_save2.h5')           # 훈련한 가중치 !!!! compile 위에!!!! / !!!! fit은 하면 안돼 !!!!
# model.summary()

# model.save(path + 'keras26_1_save.h5')

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.python.keras.callbacks import EarlyStopping
# es = EarlyStopping(
#         monitor='val_loss', mode = 'min', patience=20,
#         restore_best_weights=False,
# )

# hist = model.fit(
#         x_train, y_train, 
#         epochs=1000000, batch_size=1, 
#         verbose=2, validation_split=0.2,
#         callbacks=[es],
# )

model = load_model(path + 'keras27_mcp3_save_model.h5')     # #2와 #3 가 저장 됨

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('###################')
print('RMSE :', rmse)   
print('R2 :', r2)

# ###################
# RMSE : 4.973657451803047
# R2 : 0.7477821026080721