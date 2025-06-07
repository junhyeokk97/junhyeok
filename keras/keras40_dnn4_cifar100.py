from tensorflow.keras.datasets import cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D ,Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import time
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)
print(x_train.shape, x_test.shape)  # (50000, 3072) (10000, 3072)

y_train = pd.get_dummies(y_train.reshape(-1))
y_test = pd.get_dummies(y_test.reshape(-1))
print(y_train.shape, y_test.shape)  # (50000, 100) (10000, 100)

model = Sequential()
model.add(Dense(4000, input_dim=3072, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1000, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=250, activation='relu'))
model.add(Dense(units=100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=2,
                   restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=2,
          validation_split=0.2, callbacks=[es])
end = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('time: ', end-start)
print('loss: ', loss[0])
print('acc: ', loss[1])

# time:  62.017587184906006
# loss:  4.605241298675537
# acc:  0.009999999776482582