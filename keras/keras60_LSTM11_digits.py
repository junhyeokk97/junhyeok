from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, BatchNormalization,Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

datasets = load_digits()
x = datasets.data
y = datasets.target


# x = x.reshape(,)
x = x.reshape(1797,64,1)
y = y.reshape(1797,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50)


model = Sequential()
model.add(LSTM(55, input_shape=(64,1)))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=2,
                   restore_best_weights=True)
str = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
          verbose=2)
end = time.time()


loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('loss: ', loss[0])
print('acc: ', loss[1])
print('걸린시간: ', end-str)

# loss:  0.780370831489563
# acc:  0.20555555820465088

# loss:  1.0710406303405762
# acc:  0.21111111342906952
# 걸린시간:  47.485793352127075