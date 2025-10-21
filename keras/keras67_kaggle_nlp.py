import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

path = './_data/kaggle/nlp/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

def merge_text(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'] + 'keyword' + df['keyword'] + 'location' + df['location']
    return df

train_csv = merge_text(train_csv)
test_csv = merge_text(test_csv)

tk = Tokenizer()
tk.fit_on_texts(train_csv['text'].tolist() + test_csv['text'].tolist())
# print(tk.word_index)

x = tk.texts_to_sequences(train_csv['text'])
test_x = tk.texts_to_sequences(test_csv['text'])

y = train_csv['target']


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=57,)


print('최대길이: ', max(len(i) for i in x_train))  # 36
print('최소길이: ', min(len(i) for i in x_train))  # 1
print('평균길이: ', sum(map(len, x_train))/len(x_train))    #  18.05356882206977

vocab_size = len(tk.word_index) + 1

x_train = pad_sequences(
            x_train,
            maxlen=1000,
)

x_test = pad_sequences(
            x_test,
            maxlen=1000,
)

test_x = pad_sequences(
            test_x,
            maxlen=1000)

print(x_train.shape)    # (6851, 100)
print(x_test.shape)     # (762, 100)
# print(y_train.shape)    # (6851,)
# print(y_test.shape)     # (762,)

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train.shape)    # (6851, 1)
print(y_test.shape)     # (762, 1)




model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=20, input_length=1000))
model.add(LSTM(80))
model.add(Dense(35, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=28, restore_best_weights=True)

model.fit(x_train, y_train, epochs=1000, batch_size=250, verbose=1, callbacks=[es],
          validation_split=0.2)

loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
y_predict = (y_predict > 0.5).astype(int)
f1 = f1_score(y_test, y_predict)

f1 = f1_score(y_test, y_predict)
print('loss: ', loss)
print('f1: ', f1)

y_submit = model.predict(test_x)
y_submit = (y_submit > 0.5).astype(int).reshape(-1)

sub_csv['target'] = y_submit
sub_csv.to_csv(path + 'submission13.csv', index=False)
print('제출 파일 저장 완료:', path + 'submission11.csv')