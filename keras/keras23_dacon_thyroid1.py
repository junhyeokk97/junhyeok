# https://dacon.io/competitions/official/236488/overview/description

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd
import time
import random as rd
import matplotlib.pyplot as plt

path = './_data/dacon/thyroid/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
'''
# print(train_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
#        'Cancer'],
#       dtype='object')
# print(test_csv.columns)
# Index(['Age', 'Gender', 'Country', 'Race', 'Family_Background',
#        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk',
#        'Diabetes', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result'],
#       dtype='object')
# print(train_csv.isna().sum())   # 결측치 없음
# print(test_csv.isna().sum())    # 결측치 없음
'''
C = ['Family_Background','Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk','Diabetes']
OE = OrdinalEncoder()
train_csv[C] = OE.fit_transform(train_csv[C])
test_csv[C] = OE.transform(test_csv[C])
'''
# print(train_csv['Gender'].value_counts())
# 0.0    52248
# 1.0    34911
# print(train_csv['Country'].value_counts())
# 4.0    16309
# 1.0    13367
# 7.0    13217
# 8.0     8869
# 0.0     8856
# 5.0     7005
# 6.0     6219
# 3.0     4484
# 9.0     4451
# 2.0     4382
# print(train_csv['Race'].value_counts())
# 2.0    27090
# 1.0    20363
# 0.0    17155
# 3.0    13471
# 4.0     9080
# print(train_csv['Family_Background'].value_counts())
# 0.0    62430
# 1.0    24729
# print(train_csv['Radiation_History'].value_counts())
# 1.0    74761
# 0.0    12398
# print(train_csv['Iodine_Deficiency'].value_counts())
# 1.0    66352
# 0.0    20807
# print(train_csv['Smoke'].value_counts())
# 0.0    69889
# 1.0    17270
# print(train_csv['Weight_Risk'].value_counts())
# 0.0    61038
# 1.0    26121
# print(train_csv['Diabetes'].value_counts())
# 0.0    69736
# 1.0    17423
'''
# train_csv.to_csv(path + '확인용.csv', index=False)    #Ordinal 잘됨
# C2 = ['Gender']
# print(train_csv['Gender'].shape)    #(87159,)
# train_csv['Gender'] = train_csv['Gender'].reshape(-1,1)
# print(train_csv['Gender'].shape)
gender_col_trn = pd.get_dummies(train_csv['Gender'], prefix='Gender')
gender_col_tst = pd.get_dummies(test_csv['Gender'], prefix='Gender')
train_csv = pd.concat([train_csv.drop(columns='Gender'), gender_col_trn], axis=1)
test_csv = pd.concat([test_csv.drop(columns='Gender'), gender_col_tst], axis=1)
# country_col_trn = pd.get_dummies(train_csv['Country'], prefix='Country')
# country_col_tst = pd.get_dummies(test_csv['Country'], prefix='Country')
# train_csv = pd.concat([train_csv.drop(columns='Country'), country_col_trn], axis=1)
# test_csv = pd.concat([test_csv.drop(columns='Country'), country_col_tst], axis=1)
race_col_trn = pd.get_dummies(train_csv['Race'], prefix='Race')
race_col_tst = pd.get_dummies(test_csv['Race'], prefix='Race')
train_csv = pd.concat([train_csv.drop(columns='Race'), race_col_trn], axis=1)
test_csv = pd.concat([test_csv.drop(columns='Race'), race_col_tst], axis=1)

test_csv = test_csv.drop(['Country'], axis=1)
x = train_csv.drop(['Country','Cancer'], axis=1)
y = train_csv['Cancer']
# print(x.shape)  #(87159, 14)
# print(y.shape)  #(87159,)
r = rd.randint(1, 10000)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, shuffle=True, random_state=r, stratify=y
)

MMS = MinMaxScaler()
MMS.fit(x_train[['Age']])
x_train[['Age']] = MMS.transform(x_train[['Age']])
x_test[['Age']] = MMS.transform(x_test[['Age']])
test_csv[['Age']] = MMS.transform(test_csv[['Age']])

C1 = ['Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
SS = StandardScaler()
SS.fit(x_train[C1])
x_train[C1] = SS.transform(x_train[C1])
x_test[C1] = SS.transform(x_test[C1])
test_csv[C1] = SS.transform(test_csv[C1])

# print(x_train)  #[78443 rows x 18 columns]
# print(y_train)

# exit()
model = Sequential([
    Dense(128, input_dim=18, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.45),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(12, activation='relu'),
    Dense(1, activation='sigmoid')
])

# model.summary()

# exit()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min',
                   patience=300, restore_best_weights=True)

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
cw = dict(enumerate(class_weights))

from tensorflow.keras.callbacks import ReduceLROnPlateau
rl = ReduceLROnPlateau(monitor='val_loss', patience=300, factor=0.5, verbose=1)

from tensorflow.keras.callbacks import ModelCheckpoint
mc = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

str_time = time.time()
hist = model.fit(x_train, y_train, epochs=100000, batch_size=256,
                 verbose=2, validation_split=0.15, callbacks=[es, rl, mc],
                 class_weight=cw)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
results = np.round(results)
f1 = f1_score(y_test, results)
print('RanN:', r)
print('BCE :', np.round(loss[0], 4))
print('ACC :', np.round(loss[1], 4))
print('time:', np.round(end_time - str_time, 1))
print('F1  :', np.round(f1, 4))


y_submit = np.round(model.predict(test_csv))
submission_csv['Cancer'] = y_submit
submission_csv.to_csv(path + 'submission_0530_1.csv', index=False)
exit()
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('갑상선암')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid()
plt.show()

# submission_0.csv
# RanN: 6385
# BCE : 0.4979
# ACC : 0.8864
# time: 161.8
# F1  : 0.0407

# submission_1.csv #최고점
# RanN: 4386
# BCE : 0.3051
# ACC : 0.8804
# time: 65.5
# F1  : 0.4539

# submission_2.csv 
# RanN: 6385
# BCE : 0.2857
# ACC : 0.8949
# time: 124.6
# F1  : 0.4742

# submission_0530_0.csv
# RanN: 8676
# BCE : 0.5391
# ACC : 0.8632
# time: 188.8
# F1  : 0.385

# submission_0530_1.csv
# RanN: 7849
# BCE : 0.5607
# ACC : 0.7688
# time: 381.8
# F1  : 0.3447
