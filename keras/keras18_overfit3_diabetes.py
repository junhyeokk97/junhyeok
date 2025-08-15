from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(x.shape)  (442, 10)
# print(y.shape)  (442,)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123
)

model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
st_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=100, batch_size=32,
    verbose=2, validation_split=0.2
)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('loss :', loss)
print('RMSE :', rmse)
print('R2 :', r2)
print('걸린 시간:', end_time - st_time, '초')

plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.grid()
plt.title('보스턴')
plt.legend(loc='upper right')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# RMSE : 50.98870126067147
# R2 : 0.5873367235371451