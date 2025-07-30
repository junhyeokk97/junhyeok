import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Flatten, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
print(x_train.shape, x_test.shape) # (60000, 784) (10000, 784)

print(y_train.shape, y_test.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(-1, 1)
print(y_train.shape, y_test.shape)

y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

model = Sequential()
model.add(Dense(64, input_dim=784, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='auto',
              patience=10, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=2, callbacks=[es])
end = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('time: ', end-start)
print('loss: ', loss[0])
print('acc: ', loss[1])

# time:  481.7690`1054382324
# loss:  0.4754449725151062
# acc:  0.8330000042915344`