# https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.utils import to_categorical


path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'samplesubmission.csv')

x = train_csv.drop(columns=['target'], axis=1)
y = train_csv['target']

print(x.shape)  # (61878, 93)
print(y.shape)  # (61878,)

ohe = OneHotEncoder()
y = np.array(y)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
y = y.toarray()


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50)

Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)
test_csv = Scaler.transform(test_csv)


model = Sequential()
model.add(Dense(100, activation='relu', input_dim=93))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
model.add(Dense(9, activation= 'softmax'))

es = EarlyStopping(monitor= 'val_loss', patience=25,
                   restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam',)

model.fit(x_train, y_train, epochs=100, batch_size=1000,
          verbose=2, validation_split=0.2, callbacks=[es])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
r2 = r2_score(y_test, y_predict)

print('loss: ', loss)
print('r2: ', r2)

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
submission_csv['target']
submission_csv.to_csv(path + 'submission_1.csv', index=False)