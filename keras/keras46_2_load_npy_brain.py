import numpy as np
import pandas as pd

###### preparing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

    ### load saved data
D_path = './_data/image/'
x = np.load(D_path + 'k46_brain_x_tr.npy')
y = np.load(D_path + 'k46_brain_y_tr.npy')
xx = np.load(D_path + 'k46_brain_x_ts.npy')
yy = np.load(D_path + 'k46_brain_y_ts.npy')

# print(np.unique(yy, return_counts = True))

###### model
    ### modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(8,(4,4), input_shape = (100,100,3), activation = 'relu'))
model.add(Conv2D(16,(4,4)))
model.add(Conv2D(32,(4,4)))
model.add(Conv2D(64,(8,8)))
model.add(Conv2D(64,(16,16)))
model.add(Conv2D(64,(16,16)))
model.add(Conv2D(32,(8,8)))
model.add(MaxPool2D())
model.add(Conv2D(8,(4,4)))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

###### compile / fit
    ### compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

    ### fit
hist = model.fit(x, y, epochs = 100, batch_size = 64, 
                validation_split = 0.1, callbacks = [])

###### evaluate / predict
from sklearn.metrics import r2_score

loss = model.evaluate(xx,yy)
pred = model.predict(xx)
print(pred)
r2 = r2_score(yy, np.round(pred))
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


