# https://dacon.io/competitions/official/236068/leaderboard

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']

r = 50
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=r
)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


input = Input(shape=(8,))
dense1 = Dense(70)(input)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(100)(drop1)
dense3 = Dense(50)(dense2)
drop2 = Dropout(0.1)(dense3)
dense4 = Dense(25)(drop2)
dense5 = Dense(10)(dense3)
output = Dense(1)(dense5)
model = Model(inputs= input, outputs= output)
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 8)]               0
#  dense (Dense)               (None, 10)                90
#  dropout (Dropout)           (None, 10)                0
#  dense_1 (Dense)             (None, 10)                110
#  dense_2 (Dense)             (None, 10)                110
#  dropout_1 (Dropout)         (None, 10)                0
#  dense_3 (Dense)             (None, 10)                110
#  dense_4 (Dense)             (None, 10)                110
# =================================================================
# Total params: 530
# Trainable params: 530
# Non-trainable params: 0

es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=100, restore_best_weights=True
)


import datetime
date = datetime.datetime.now()
print(date)                     
print(type(date))               
date = date.strftime('%m%d_%H%M')
print(date)                    
print(type(date))              

path1 = './_save/dacon/thyroid/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path1, 'k28_', date, '_', filename])

from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto',
    save_best_only=True, 
    filepath=filepath
)

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
start_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=1000, batch_size=55,
    verbose=2, validation_split=0.1,
    callbacks=[es,]                  # ''는 값 / 없으면 변수
)
end_time = time.time()

results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
accuracy_score = accuracy_score(y_test, y_predict)

print('Loss   :', round(results[0], 4))
print('Acc    :', round(accuracy_score, 4))

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
submission_csv['Outcome'] = y_submit
submission_csv.to_csv(path + 'submission_15.csv', index=False)


# Loss   : 0.5203
# Acc    : 0.7879

# MinMaxScaler
# Loss   : 0.3736
# Acc    : 0.8333

# MaxAbsScaler
# Loss   : 0.5119
# Acc    : 0.8485

# StandardScaler
# Loss   : 0.4167
# Acc    : 0.8485

# RobustScaler
# Loss   : 0.4021
# Acc    : 0.8182