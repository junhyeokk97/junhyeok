from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=1000,
    test_split=0.2,
    maxlen=200 # 최대 단어 길이가 100개까지 있는 문장.
)

print(x_train)
print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,) >> ,100
print(y_train[0])   # 3
print(np.unique(y_train))  
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

print('뉴스기사 최대길이: ', max(len(i) for i in x_train))  # 2376
print('뉴스기사 최소길이: ', min(len(i) for i in x_train))  # 13
print('뉴스기사 평균길이: ', sum(map(len, x_train))/len(x_train))   # 145.5398574927633

xtr = pad_sequences(
            x_train,
            maxlen=100,
)

xts = pad_sequences(
            x_test,
            maxlen=100,
)

print(xtr.shape)    # (7076, 100)
print(xts.shape)    # (1770, 100)
print(y_train.shape)    # (7076,)
print(y_test.shape)    # (1770,)
y_train = np.array(y_train)
y_test = np.array(x_test)

y_train = y_train.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(xtr)
y_train = y_train.reshape(-1, 111, 9)
print(y_train.shape)    # (707600, 111, 9)

y_test = y_test.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
y_test = ohe.fit_transform(xts)
y_test = y_test.reshape(-1, 111, 9)
print(y_test.shape)    # (177000, 111, 9)



model = Sequential()
model.add(Embedding(111, 9))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)

model.fit(xtr, y_train, epochs=100, batch_size=100, verbose=2, callbacks=[es],
          validation_split=0.2)

loss = model.evaluate(xts, y_test)
y_predict = model.predict(xts)
print('loss: ', loss)
print('answer: ', y_predict)
