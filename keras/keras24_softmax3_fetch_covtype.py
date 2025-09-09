import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]),
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510]

y = pd.get_dummies(y)
print(y)
print(y.shape)  # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=15,
                                                    stratify=y)

model = Sequential()
model.add(Dense(100, input_dim=54, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
model.add(Dense(7, activation='softmax'))       

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])                                            

es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8500, verbose=2,
          validation_split=0.2, callbacks=[es])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)

print('loss: ', loss[0])
print('acc: ', loss[1])



# loss:  0.23000773787498474
# acc:  0.9084885120391846