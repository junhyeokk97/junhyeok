from tensorflow.keras.datasets import cifar10
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

y_train = pd.get_dummies(y_train.reshape(-1))
y_test = pd.get_dummies(y_test.reshape(-1))
print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)


model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, input_shape=(32,32,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=12))
model.add(Dense(units=10, activation='softmax'))
# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=2,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=2,
          validation_split=0.2, callbacks=[es])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test.to_numpy(), axis=1)
acc = accuracy_score(y_test, y_predict)

print('loss: ', loss[0])
print('acc: ', loss[1])

import matplotlib.pyplot as plt
plt.imshow(x_train, 'gray')
plt.show()