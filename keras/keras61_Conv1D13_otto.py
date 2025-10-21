import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, LSTM, BatchNormalization,Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'samplesubmission.csv')

x = train_csv.drop(columns=['target'], axis=1)
y = train_csv['target']

print(x.shape)  # (61878, 93)
print(y.shape)  # (61878,)

x = x.values.reshape(x.shape[0], 31, 3)
y = y.values.reshape(y.shape[0], 1)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50)

# print(x_train.shape, x_test.shape) # (55690, 93) (6188, 93)
# x_train = x_train.values.reshape(55690, 31, 3, 1)
# x_test = x_test.values.reshape(6188, 31, 3 ,1)
# print(x_train.shape, x_test.shape) # (55690, 31, 3, 1) (6188, 31, 3, 1)

model = Sequential()
model.add(Conv1D(32, 2, input_shape=(31, 3)))
model.add(Conv1D(31, 30))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(9, activation='softmax'))



es = EarlyStopping(monitor='val_loss', patience=30,
                   restore_best_weights=True)

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['categorical_accuracy'])

str = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=1000,
          verbose=2, validation_split=0.2,
          callbacks=[es])
end = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_class = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, y_predict)

print('loss: ', loss[0])
print('acc: ', acc)
print('걸린시간: ', end-str)

# loss:  0.5578269958496094
# acc:  0.7582417582417582

# loss:  0.5578269958496094
# acc:  0.7582417582417582