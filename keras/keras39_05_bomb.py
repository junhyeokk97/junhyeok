from tensorflow.keras.datasets import cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D ,Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)


y_train = pd.get_dummies(y_train.reshape(-1))
y_test = pd.get_dummies(y_test.reshape(-1))
print(y_train.shape, y_test.shape)   # (50000, 100) (10000, 100)

model = Sequential()
model.add(Conv2D(1000, (2,2), strides=1, input_shape=(32,32,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=2000, kernel_size=(2,2)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(1000, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=11000))
model.add(Dense(units=100, activation='softmax'))
# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=50,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1,
          validation_split=0.2, callbacks=[es])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test.to_numpy(), axis=1)
acc = accuracy_score(y_test, y_predict)

print('loss: ', loss[0])
print('acc: ', loss[1])
