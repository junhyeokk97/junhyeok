# 1. train_csv에서 casual과 registered를 y로 잡는다.
# 2. 훈련을 진행하고, test_csv의 casual과 registered를 예측한다.
# 3. 예측한 casual과 registered를 test_csv에 추가한다. 
#    (N, 8) -> (N, 10) new_test.csv파일을 만든다.

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

# 1. Data
path = './_data/kaggle/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

X = train_csv.drop(columns=['count', 'casual', 'registered'], axis=1)
# y = train_csv[['casual', 'registered', 'count',]]
y = train_csv[['casual', 'registered']]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=0,
)

# 2. Model Configuration
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=8)) # input_dim 부분이 input_layer
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='relu'))

# 3. Compile, Training
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=16)

# 4. Evaluation, Predict
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE : {rmse}")
print(f"R2 : {r2}")

y_submit = model.predict(test_csv)
print(type(y_submit)) # tensorflow model의 출력값은 <class 'numpy.ndarray'>
# pandas의 데이터 부분은 numpy array로 저장되기에 호환이 가능

# 데이터 원본 유지를 위한 사본 만들기
test_csv_copy = test_csv.copy(deep=True) # deep=True가 기본값
test_csv_copy[['casual', 'registered']] = y_submit
test_csv_copy.to_csv(path + 'new_test.csv')
