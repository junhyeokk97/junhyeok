from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=100000,
                                                      )
print(x_train)
print(y_train)
print(x_train.shape)    # (25000, )
print(y_train.shape)    # (25000, )
print(np.unique(y_train, return_counts=True))
# (array([0,1], dtype=int64), array([12500, 12500], dtype=int64))
# print(pd.value)

print('영화평 최대길이: ', max(len(i) for i in x_train))  # 2494
print('영화평 최소길이: ', min(len(i) for i in x_train))  # 11
print('영화평 평균길이: ', sum(map(len, x_train))/len(x_train))   # 238.71364

# 실습
x_train = pad_sequences(x_train,
                        maxlen=1000)

x_test = pad_sequences(x_test,
                       maxlen=1000)

print(x_train)
print(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(x_train.shape)    # (25000, 200)
print(y_train.shape)    # (25000, 1)
print(x_test.shape)    # (25000, 200)
print(y_test.shape)    # (25000, 1)


# ohe = OneHotEncoder(sparse=False)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.fit_transform(y_test)

# x_train = x_train.reshape(-1, 100, 2)
# x_test = x_test.reshape(-1, 100, 2)

model = Sequential()
model.add(Embedding(100000, 100))
model.add(LSTM(64))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss',
                    mode='min', patience=15, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=2,
          callbacks=es)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = (y_predict > 0.5).astype(int)
print('loss: ', loss)
print('answer: ', y_predict)
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

# loss:  [0.2937597930431366]
# acc:  0.87956