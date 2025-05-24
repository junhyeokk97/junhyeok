import numpy as np
import pandas as pd
import time
###### preparing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
    ### load saved data

D_path = './_data/tensor_cert/horsehuman/'
x_tr = np.load(D_path + 'k46_horse_x_tr.npy')
y_tr = np.load(D_path + 'k46_horse_y_tr.npy')
x_ts = np.load(D_path + 'k46_horse_x_ts.npy')
y_ts = np.load(D_path + 'k46_horse_y_ts.npy')

###### model
    ### modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, MaxPool2D, Flatten, Dropout, BatchNormalization

x_tr = x_tr.reshape(x_tr.shape[0], 10000, 3)
y_tr = y_tr.reshape(y_tr.shape[0], 1)
x_ts = x_ts.reshape(x_ts.shape[0], 10000, 3)
y_ts = y_ts.reshape(y_ts.shape[0], 1)

print(x_tr.shape, y_tr.shape)   # (1027, 10000, 3) (1027, 1)

model = Sequential()
model.add(LSTM(50, input_shape=(10000, 3)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

###### compile / fit
    ### compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

str = time.time()
    ### fit
hist = model.fit(x_tr, y_tr, epochs = 100, batch_size = 64, 
                validation_split = 0.1, callbacks = [])
end = time.time()

    ### save
# model.save(D_path + 'model.hdf5')

###### evaluate / predict
from sklearn.metrics import r2_score

loss = model.evaluate(x_ts, y_ts)
pred = model.predict(x_ts)
r2 = r2_score(y_ts, np.round(pred))
print('loss : ', loss[0])
print('acc : ', loss[1])
print('r2 : ', r2)
print('걸린시간: ', end-str)
# ### plotting history
# import matplotlib.pyplot as plt

# fig, ax1 = plt.subplots()
# ax1.plot(hist.history['loss'],c = 'red', label = 'loss')
# ax1.plot(hist.history['val_loss'],'r--', label = 'val_loss')
# ax1.set_xlabel('epochs')
# ax1.set_ylabel('loss')
# ax1.legend(loc = "best")

# ax2 = ax1.twinx()
# ax2.plot(hist.history['acc'],'b', label = 'acc')
# ax2.plot(hist.history['val_acc'],'b--', label = 'val_acc')
# ax2.set_ylabel('acc')
# ax2.legend(loc = 'best')

# plt.grid()
# plt.show()


