# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTENC

#1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv')


# print(train_csv.head()) # 위쪽 행만 보기 디폴트값=5
# print(train_csv.tail()) # 아래쪽 행만 보기 디폴트값=5
# print(train_csv.head(10)) # 위쪽 10개 행만 보기

print(train_csv.isna().sum())
print(test_csv.isna().sum())

print(train_csv.columns)        # ['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
                                #  'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                                #  'EstimatedSalary', 'Exited']

# 문자 데이터 수치화
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
print(y.value_counts())

smotenc = SMOTENC(random_state=50,
                  categorical_features=[2,3,6,7,8])



x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=102)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train)
class_weight_dict = dict(enumerate(class_weights))


print(x_train.shape, x_test.shape)  # (148530, 10) (16504, 10)
print(y_train.shape, y_test.shape)  # (148530,) (16504,)

model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'val_loss', mode= 'min',
                   patience= 32, restore_best_weights=True, min_delta=1e-4)

model.fit(x_train, y_train, epochs=1000, batch_size=250, verbose=2,
          validation_split=0.2, callbacks=[es], class_weight=class_weight_dict)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# print(y_predict)

y_predict = np.round(y_predict)

acc_score = accuracy_score(y_test, y_predict)
print('loss: ', loss[0])
print('acc_score: ', loss[1])
y_submit = model.predict(test_csv)

submission_csv['Exited'] = y_submit

submission_csv.to_csv( path + 'submissionss.0529.1644.csv', index=False)



