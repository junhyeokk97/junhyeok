# https://dacon.io/competitions/official/236068/leaderboard

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization,Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler


path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=50)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (521, 8) (131, 8)

x_train = x_train.reshape(521, 4, 2, 1)
x_test = x_test.reshape(131, 4, 2, 1)
print(x_train.shape, x_test.shape)  # (521, 4, 2, 1) (131, 4, 2, 1)

model = Sequential()
model.add(Conv2D(32, (2,2), strides=1, input_shape=(4, 2, 1),padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(2,2),padding='same', activation='relu'))
model.add(Conv2D(16, (2,2),padding='same' , activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=2,
                   restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
          verbose=2, callbacks=[es])

results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
print('[BCE] :', round(results[0], 4))
print('[ACC] :', round(results[1], 4))

# [BCE] : 0.5665
# [ACC] : 0.7099