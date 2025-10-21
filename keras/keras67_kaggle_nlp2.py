import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import RobustScaler, LabelEncoder
from keras.layers.merge import concatenate , Concatenate


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

x1 = train_csv['keyword']
x2 = train_csv['location']
x3 = train_csv['text']
x1_submit = test_csv['keyword']
x2_submit = test_csv['location']
x3_submit = test_csv['text']

y = train_csv['target']

tk = Tokenizer()
tk.fit_on_texts(x1)
x1 = tk.texts_to_sequences(x1)
x1 = pad_sequences(x1, maxlen=100,)

x1_submit = tk.texts_to_sequences(x1_submit)
x1_submit = pad_sequences(x1_submit, maxlen=100)


x2 = tk.texts_to_sequences(x2)
x2 = pad_sequences(x2, maxlen=100,)

x2_submit = tk.texts_to_sequences(x2_submit)
x2_submit = pad_sequences(x2_submit, maxlen=100)


x3 = tk.texts_to_sequences(x3)
x3 = pad_sequences(x3, maxlen=100,)

x3_submit = tk.texts_to_sequences(x3_submit)
x3_submit = pad_sequences(x3_submit, maxlen=100)



x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
                                                    x1,x2,x3,y,
                                                    test_size=0.15,
                                                    random_state=87,)

print(x1_train.shape, x1_test.shape)    # (6090,) (1523,)
print(x2_train.shape, x2_test.shape)    # (6090,) (1523,)
print(x3_train.shape, x3_test.shape)    # (6090, 1000) (1523, 1000)

y_train = np.array(y_train)
y_test = np.array(y_test)

# x1_train = np.array(x1_train)
# x1_test = np.array(x1_test)
# x2_train = np.array(x2_train)
# x2_test = np.array(x2_test)


# x1_train = x1_train.reshape(-1, 1)
# x2_train = x2_train.reshape(-1, 1)
# x2_test = x2_test.reshape(-1, 1)
# x2_test = x2_test.reshape(-1, 1)
# x1_submit = np.array(x1_submit).reshape(-1, 1)
# x2_submit = np.array(x2_submit).reshape(-1, 1)


y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(y_train.shape)    # (6090, 1)
print(y_test.shape)     # (1523, 1)

print(x1_train.shape)
print(x2_train.shape)
print(x3_train.shape)

vocab_size = len(tk.word_index) + 1

input1 = Input(shape=(100,))
dense1 = Embedding(input_dim=vocab_size, output_dim=20, input_length=100)(input1)
flat = Flatten()(dense1)
dense2 = Dense(20, activation='relu')(flat)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(40, activation='relu')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model1 = Model(inputs = input1, outputs=output1)

input2 = Input(shape=(100,))
dense21 = Embedding(input_dim=vocab_size, output_dim=20, input_length=100)(input2)
flat = Flatten()(dense21)
dense22 = Dense(20, activation='relu')(flat)
dense23 = Dense(30, activation='relu')(dense22)
dense24 = Dense(40, activation='relu')(dense23)
output2 = Dense(1, activation='sigmoid')(dense24)
model2 = Model(inputs = input2, outputs=output2)



input3 = Input(shape=(100,))
dense31 = Embedding(input_dim=vocab_size, output_dim=20, input_length=100)(input3)
flat = Flatten()(dense31)
dense32 = Dense(20, activation='relu')(flat)
dense33 = Dense(30, activation='relu')(dense32)
dense34 = Dense(40, activation='relu')(dense33)
output3 = Dense(1, activation='sigmoid')(dense34)
model3 = Model(inputs = input3, outputs=output3)


merge1 = concatenate([output1, output2, output3])
merge2 = Dense(10)(merge1)
merge3 = Dense(10)(merge2)
merge4 = Dense(10)(merge3)
last_output = Dense(1, activation='sigmoid')(merge4)
model = Model(inputs=[input1, input2, input3], outputs=last_output)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=14, restore_best_weights=True)

model.fit([x1_train, x2_train, x3_train], y_train, epochs=1000, batch_size=170,
          verbose=2, callbacks=[es], validation_split=0.2)

loss = model.evaluate([x1_test,x2_test,x3_test], y_test)

y_predict = model.predict([x1_test,x2_test,x3_test])
y_predict = (y_predict > 0.5).astype(int)
f1 = f1_score(y_test, y_predict)

print('loss: ', loss)
print('f1: ', f1)

y_submit = model.predict([x1_submit, x2_submit, x3_submit])
y_submit = (y_submit > 0.5).astype(int).reshape(-1)


sub_csv['target'] = y_submit
sub_csv.to_csv(path + 'submission04.csv', index=False)
print('제출 파일 저장 완료:', path + 'submission00.csv')