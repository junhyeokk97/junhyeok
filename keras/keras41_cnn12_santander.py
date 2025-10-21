import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization,Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

path = './_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(columns=['target'], axis=1)
y = train_csv['target']

# print(x.shape)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50)

print(x_train.shape, x_test.shape) # (180000, 200) (20000, 200)
x_train = x_train.values.reshape(180000, 20, 10, 1)
x_test = x_test.values.reshape(20000, 20, 10 ,1)
print(x_train.shape, x_test.shape) # (180000, 20, 10, 1) (20000, 20, 10, 1)


model = Sequential()
model.add(Conv2D(32, (3,3), strides=1, input_shape=(20, 10, 1),padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3),padding='same', activation='relu'))
model.add(Conv2D(16, (2,2),padding='same' , activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



es = EarlyStopping(monitor='val_loss', patience=30,
                   restore_best_weights=True)

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=1000,
          verbose=2, validation_split=0.2,
          callbacks=[es])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)


print('loss: ', loss[0])
print('acc: ', acc)