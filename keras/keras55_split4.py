import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

a = np.array(range(1,101))
x_predict = np.array(range(96,106))     # 101 ~ 107 찾기

timesteps = 11       # x = n, 10, 1   >>  n, 5, 2
                     # y = n, 1

print(a.shape)  # (100,)

def split_x(dataset, timesteps):
    aa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i+timesteps)]
        aa.append(subset)
    return np.array(aa)

m = split_x(a, timesteps=timesteps)

x = m[:, :-1].reshape(-1,5,2)
y = m[:, -1]
print(x.shape, y.shape) # (90, 10) (90,)

# x = x.reshape(x.shape[0], x.shape[1], 1)
# print(x.shape)  # (90, 10, 1)

model = Sequential()
model.add(LSTM(27, input_shape=(5,2)))
model.add(Dense(25, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000,verbose=2)

loss = model.evaluate(x,y)
print('loss: ', loss)
x_predict = x_predict.reshape(-1, 5, 2)
y_predict = model.predict(x_predict)
print('p: ', y_predict)


