import numpy as np
import pandas as pd

###### preparing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

    ### load saved data

D_path = './_data/kaggle/tensor_cert/horse-or-human/'
x_tr = np.load(D_path + 'k46_horse_x_tr.npy')
y_tr = np.load(D_path + 'k46_horse_y_tr.npy')
x_ts = np.load(D_path + 'k46_horse_x_ts.npy')
y_ts = np.load(D_path + 'k46_horse_y_ts.npy')

###### model
    ### modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

model = Sequential()
model.add(Conv2D(8,(8,8), input_shape = (100,100,3), activation = 'relu'))
model.add(Conv2D(16,(8,8)))
model.add(Conv2D(64,(16,16)))
model.add(MaxPool2D())
model.add(Conv2D(64,(16,16)))
model.add(MaxPool2D())
model.add(Conv2D(8,(4,4)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(256))
model.add(Dense(4))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

###### compile / fit
    ### compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

    ### fit
hist = model.fit(x_tr, y_tr, epochs = 100, batch_size = 64, 
                validation_split = 0.1, callbacks = [])

    ### save
model.save(D_path + 'model.hdf5')

###### evaluate / predict
from sklearn.metrics import r2_score

loss = model.evaluate(x_ts, y_ts)
pred = model.predict(x_ts)
r2 = r2_score(y_ts, np.round(pred))
print('loss : ', loss[0])
print('acc : ', loss[1])
print('r2 : ', r2)

### plotting history
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.plot(hist.history['loss'],c = 'red', label = 'loss')
ax1.plot(hist.history['val_loss'],'r--', label = 'val_loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend(loc = "best")

ax2 = ax1.twinx()
ax2.plot(hist.history['acc'],'b', label = 'acc')
ax2.plot(hist.history['val_acc'],'b--', label = 'val_acc')
ax2.set_ylabel('acc')
ax2.legend(loc = 'best')

plt.grid()
plt.show()


