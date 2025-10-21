
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (581012, 54) (581012,)
# print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]),
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510]

y = pd.get_dummies(y)
# print(y)
# print(y.shape)  # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=42)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = StandardScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(100, input_dim=54, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])                                            

es = EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True)


# import datetime
# date = datetime.datetime.now()
# print(date)     # 2025-06-02 13:00:40.340100
# print(type(date))   # <class 'datetime.datetime'>
# date = date.strftime('%m%d_%H%M')
# print(date) # 0602_1306
# print(type(date))   # <class 'str'>   str = 문자열

# path = './_save/keras28_mcp/10_fetch_covtype/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # epoch 앞 4자리, val_loss 소수점 4자리까지
# filepath = "".join([path, 'k28_', date, '_', filename])

# print(filepath)

# mcp = ModelCheckpoint(monitor= 'val_loss', mode= 'auto', save_best_only=True,
#                       filepath=filepath, save_weights_only=True)

model.fit(x_train, y_train, epochs=100, batch_size=1500, verbose=2, validation_split=0.1, 
                 callbacks=[es])

# path = './_save/keras28_mcp/10_fetch_covtype/'
# model = load_model(path + 'keras28_covtype_save.h5')



#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
y_pred = (y_predict > 0.5).astype(int)

acc_score = accuracy_score(y_test, y_predict)

print('loss: ', loss[0])
print('accuracy: ', loss[1])

# loss:  0.28779327869415283
# accuracy:  0.8798320293426514