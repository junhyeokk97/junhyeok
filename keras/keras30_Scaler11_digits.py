import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

datasets = load_digits()
x = datasets.data
y = datasets.target

print(x.shape)   # (1797, 64)
print(y.shape)   # (1797,)
print(np.max(x)) # 16.0
print(np.min(x)) # 0.0
exit()
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50)

model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=64))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer ='adam')
history = model.fit(x_test, y_test, epochs=1000, batch_size=4,
                    validation_split=0.2)

def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

results= model.predict(x_test)
rmse = RMSE(results, y_test)
r2 = r2_score(results, y_test)

print('r2_score: ', r2)
print('rmse: ', rmse)

# r2_score:  0.9571018287880121
# rmse:  0.6171279513843496