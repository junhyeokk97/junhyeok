import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

datasets = load_digits()
x = datasets.data
y = datasets.target


# x = x.reshape(,)
x = x.reshape(1797,8,8)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50)

model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(8,8)))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer ='adam')
history = model.fit(x_test, y_test, epochs=100, batch_size=4,
                    validation_split=0.2)

def RMSE (y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_predict))

results= model.predict(x_test)
# rmse = RMSE(y_test, results)
# r2 = r2_score(y_test, results)

# print('r2_score: ', r2)
# print('rmse: ', rmse)

# aaa = 10
# print(y_train[aaa])

import matplotlib.pyplot as plt
plt.imshow(x_train[0],'gray')
plt.show()

# r2_score:  0.9571018287880121
# rmse:  0.6171279513843496