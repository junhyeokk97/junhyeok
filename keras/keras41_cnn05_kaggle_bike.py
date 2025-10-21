from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import time

# 1. 데이터
path = './_data/kaggle/bike'
train_csv = pd.read_csv(path + '/train.csv', index_col=0)
test_csv =  pd.read_csv(path + '/test.csv', index_col=0)
submission_csv = pd.read_csv(path + '/sampleSubmission.csv')

x = train_csv.drop(columns=['casual','registered','count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                                    random_state=100)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=50)
print(x_train.shape, x_test.shape)  # (8708, 8) (2178, 8)

x_train = x_train.values.reshape(8708, 4, 2, 1)
x_test = x_test.values.reshape(2178, 4, 2, 1)
print(x_train.shape, x_test.shape)  # (8708, 4, 2, 1) (2178, 4, 2, 1)

model = Sequential()
model.add(Conv2D(32, (2,2), strides=1, input_shape=(4, 2, 1),padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(2,2),padding='same', activation='relu'))
model.add(Conv2D(16, (2,2),padding='same' , activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=2,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
          verbose=2)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('loss: ', loss)
print('r2: ', r2)