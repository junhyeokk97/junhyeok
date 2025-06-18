# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()     # 클래스를 정의화 한다. > 인스턴스화 한다.
le_gen = LabelEncoder()
# train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
le_geo.fit(train_csv['Geography'])
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])

le_gen.fit(train_csv['Gender'])
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])

le_geo.fit(test_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(test_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# test_csv['Geography'] = le_geo.fit_transform(test_csv['Geography'])
# test_csv['Gender'] = le_gen.fit_transform(test_csv['Gender'])

print(train_csv['Geography'])
print(train_csv['Geography'].value_counts())
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'])
print(train_csv['Gender'].value_counts())
# 1    93150
# 0    71884


train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)
print(train_csv.columns)    # ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                            #    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                            #    'Exited']

x = train_csv.drop(['Exited'], axis=1)
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)

x = x.to_numpy().reshape(x.shape[0], x.shape[1], 1)  # (165034, 10, 1)
y = y.to_numpy().reshape(y.shape[0], 1)  # (165034, 1)


print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=42)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# test_csv = scaler.transform(test_csv)

# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train)
# class_weight_dict = dict(enumerate(class_weights))



# print(x_train.shape, x_test.shape)  # (148530, 10) (16504, 10)
# print(y_train.shape, y_test.shape)  # (148530,) (16504,)

model = Sequential()
model.add(LSTM(32, input_shape=(10, 1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'val_loss', mode= 'min',
                   patience= 32, restore_best_weights=True, min_delta=1e-4)

# import datetime
# date = datetime.datetime.now()
# print(date)     # 2025-06-02 13:00:40.340100
# print(type(date))   # <class 'datetime.datetime'>
# date = date.strftime('%m%d_%H%M')
# print(date) # 0602_1306
# print(type(date))   # <class 'str'>   str = 문자열

# path = './_save/keras28_mcp/08_kaggle_bank/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # epoch 앞 4자리, val_loss 소수점 4자리까지
# filepath = "".join([path, 'k28_', date, '_', filename])

# print(filepath)

# # path = './_save/keras28_mcp/01_boston/'
# mcp = ModelCheckpoint(monitor= 'val_loss', mode= 'auto', save_best_only=True,
#                       filepath=filepath)
str = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=1000, verbose=2,
          validation_split=0.2, callbacks=[es])
end = time.time()
# path = './_save/keras28_mcp/08_kaggle_bank/'
# model.save(path + 'keras28_bank_save.h5')



#4. 평가, 예측
print("=======================================")
results = model.evaluate(x_test,y_test)

print('loss: ', round(results[0], 4))
print('acc: ', round(results[1], 5))

y_predict = model.predict(x_test)
print(y_predict[:10])
y_predict = np.round(y_predict)
print(y_predict)
print('걸린시간: ', end - str)


# loss:  0.3973
# acc:  0.82986

# loss:  0.3928
# acc:  0.82386


# loss:  0.3202
# acc:  0.8673
# 걸린시간:  182.92122149467468