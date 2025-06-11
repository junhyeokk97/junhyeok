# https://www.kaggle.com/competitions/santander-customer-transaction-prediction
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.utils import class_weight

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

Scaler = MaxAbsScaler()
Scaler.fit(x_train)
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)
test_csv = Scaler.transform(test_csv)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train)
class_weights = dict(enumerate(weights))

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=200))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
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


y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
submission_csv['target']
submission_csv.to_csv(path + 'submission_1.csv')
