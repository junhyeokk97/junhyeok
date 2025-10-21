import numpy as np
import pandas as pd

###### preparing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

    ### image to numpy
D_path = './_data/image/brain/'
train_data = ImageDataGenerator(rescale = 1./255)
test_data = ImageDataGenerator(rescale = 1./255)

train_set = train_data.flow_from_directory(D_path + 'train',
                                           target_size = (100,100),
                                           batch_size = 80,
                                           class_mode = 'binary',
                                           color_mode = 'rgb',
                                           shuffle = True
                                           )

test_set = test_data.flow_from_directory(D_path + 'test',
                                         target_size = (100,100),
                                         batch_size = 80,
                                         class_mode = 'binary',
                                         color_mode = 'rgb',
                                         shuffle = True
                                         )

# print(train_set[0][0].shape, test_set[0][0].shape) # (80, 100, 100, 3) (80, 100, 100, 3)
# print(train_set[0][1].shape, test_set[0][1].shape) # (80,) (80,)
    ### concatenate all batch
all_x = []
all_y = []
xx = []
yy = []
for i in range(len(train_set)):
    x_batch, y_batch = train_set[i]
    x_b, y_b = test_set[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
    xx.append(x_b)
    yy.append(y_b)
    

x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
xx = np.concatenate(xx, axis=0)
yy = np.concatenate(yy, axis=0)
# print('x.shape: ', x.shape) # x.shape:  (160, 100, 100, 3)
# print('y.shape: ', y.shape) # y.shape:  (160,)

    ### save data as npy
D_path = './_data/image/'
np.save(D_path + 'k46_brain_x_tr.npy', arr = x)
np.save(D_path + 'k46_brain_y_tr.npy', arr = y)
np.save(D_path + 'k46_brain_x_ts.npy', arr = xx)
np.save(D_path + 'k46_brain_y_ts.npy', arr = yy)

###### model
    ### modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(8,(8,8), input_shape = (100,100,3), activation = 'relu'))
model.add(Conv2D(16,(8,8), activation = 'relu'))
model.add(Conv2D(32,(8,8), activation = 'relu'))
model.add(Conv2D(64,(16,16), activation = 'relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,(16,16), activation = 'relu'))
model.add(Conv2D(32,(8,8), activation = 'relu'))
model.add(MaxPool2D())
model.add(Conv2D(8,(4,4), activation = 'relu'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Dropout(0.2))
model.add(Dense(4))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

###### compile / fit
    ### compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

    ### fit
hist = model.fit(x, y, epochs = 20, batch_size = 1, 
                validation_split = 0.1, callbacks = [])

###### evaluate / predict
from sklearn.metrics import r2_score

loss = model.evaluate(test_set[0][0], test_set[0][1])
pred = model.predict(test_set[0][0])
r2 = r2_score(test_set[0][1], pred)

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


