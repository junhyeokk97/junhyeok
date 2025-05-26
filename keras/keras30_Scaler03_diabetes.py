from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from tensorflow.keras.callbacks import EarlyStopping

dataset = load_diabetes()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=50
)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=False,
)


st_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=100000, batch_size=32,
    verbose=2, validation_split=0.2,
    callbacks=[es,]
)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('RMSE :', rmse)
print('R2 :', r2)


# RMSE : 50.469586873866426
# R2 : 0.542062781751149

# MinMaxScaler
# RMSE : 50.73969453556555
# R2 : 0.5371480334293155

# MaxAbsScaler
# RMSE : 49.63791010045271
# R2 : 0.5570309888342313

# StandardScaler
# RMSE : 53.714245920833235
# R2 : 0.4812891184686696

# RobustScaler
# RMSE : 52.68465016758913
# R2 : 0.500983793006675