import numpy as np
import pandas as pd

###### preparing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,BatchNormalization, Input

    ### image to numpy
D_path = './_data/kaggle/men_women/'
train_set = ImageDataGenerator(rescale = 1./255,
                               horizontal_flip = True,
                               width_shift_range = 0.2,
                               vertical_flip = True,
                               height_shift_range = 0.3)
test_set = ImageDataGenerator(rescale = 1./255)

train = train_set.flow_from_directory(D_path,
                                    target_size = (100,100),
                                    batch_size = 100,
                                    class_mode = 'binary',
                                    color_mode = 'rgb',
                                    shuffle = True,
                                    seed = 50)
test = test_set.flow_from_directory(D_path,
                                    target_size = (100,100),
                                    batch_size = 100,
                                    class_mode = 'binary',
                                    color_mode = 'rgb',
                                    shuffle = True,
                                    seed = 50)

print(train[0][0].shape, test[0][0].shape) # (100, 100, 100, 3) (100, 100, 100, 3)
print(train[0][1].shape, test[0][1].shape) # (100,) (100,)
print(len(train), len(test)) # 34 34        

    ### concatenate all batch
all_x = []
all_y = []
xx = []
yy = []
for i in range(len(train)):
    x_batch, y_batch = train[i]
    x_b, y_b = test[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
    xx.append(x_b)
    yy.append(y_b)
    
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
xx = np.concatenate(xx, axis=0)
yy = np.concatenate(yy, axis=0)
print('x.shape: ', x.shape) # x.shape:  (3309, 100, 100, 3)
print('y.shape: ', y.shape) # y.shape:  (3309,)

    ### save data as npy
np.save(D_path + 'gender_x_tr.npy', arr = x)
np.save(D_path + 'gender_y_tr.npy', arr = y)
np.save(D_path + 'gender_x_ts.npy', arr = xx)
np.save(D_path + 'gender_y_ts.npy', arr = yy)

###### model
    ### modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

model = Sequential()
model.add(Conv2D(64,(8,8), input_shape = (100,100,3), activation = 'relu'))
model.add(Conv2D(32,(8,8), activation = 'relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32,(8,8), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16,(8,8), activation = 'relu'))
model.add(Conv2D(8,(4,4), activation = 'relu'))
model.add(Flatten())
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

###### compile / fit
    ### compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

    ### fit
hist = model.fit(x,y, epochs = 20, batch_size = 128, 
                validation_split = 0.1, callbacks = [])

###### evaluate / predict
from sklearn.metrics import r2_score

loss = model.evaluate(xx, yy)
pred = model.predict(xx)
print('loss : ', loss[0])
print('acc : ', loss[1])

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

