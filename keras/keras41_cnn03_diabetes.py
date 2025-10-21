import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
import time
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import OneHotEncoder

dataset = load_diabetes()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=50
)

print(x_train.shape , x_test.shape) # (353, 10) (89, 10)
print(y_train.shape , y_test.shape) # (353,) (89,)

x_train = x_train.reshape(353, 5, 2, 1)
x_test = x_test.reshape(89, 5, 2, 1)
print(x_train.shape, x_test.shape)

model = Sequential()
model.add(Conv2D(32, (1,1), strides=1, input_shape=(5, 2, 1),padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (1,1),padding='same' , activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=2,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
          verbose=2)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('loss: ', loss[0])
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)


# loss:  2710.3828125
# r2:  0.5127217447370118