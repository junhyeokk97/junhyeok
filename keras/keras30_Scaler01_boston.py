from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import sklearn as sk

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.datasets import load_boston
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

datasets = load_boston()
print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=42)

# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = StandardScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(14, input_dim=13, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam', )

es = EarlyStopping(monitor = 'val_loss', mode = 'min',
                   patience = 15, restore_best_weights= True,)

model.fit(x_train, y_train, epochs=90, batch_size=1, verbose=2, validation_split=0.1, 
                 callbacks=[es])

print("=======================================")
loss = model.evaluate(x_test,y_test)
results = model.predict(x_test)
print("loss : ", loss)

# loss :  7.830817699432373