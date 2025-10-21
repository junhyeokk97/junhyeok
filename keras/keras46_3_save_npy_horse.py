import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
###### preparing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

    ### image to numpy
D_path = './_data/tensor_cert/horsehuman/'
train_set = ImageDataGenerator(rescale = 1./255,
                               horizontal_flip = True,
                               vertical_flip = True,
                               width_shift_range = 0.2,
                               height_shift_range = 0.2)
test_set = ImageDataGenerator(rescale = 1./255)

train = train_set.flow_from_directory(D_path,
                                    target_size = (100,100),
                                    batch_size = 1027,
                                    class_mode = 'binary',
                                    color_mode = 'rgb',
                                    shuffle = True,
                                    seed = 32)
test = test_set.flow_from_directory(D_path,
                                    target_size = (100,100),
                                    batch_size = 1027,
                                    class_mode = 'binary',
                                    color_mode = 'rgb',
                                    shuffle = True,
                                    seed = 67)

# print(train[0][0].shape, test[0][0].shape) # (80, 100, 100, 3) (80, 100, 100, 3)
# print(train[0][1].shape, test[0][1].shape) # (80,) (80,)
# print(len(train), len(test)) # 13 13

#     ### concatenate all batch
# all_x = []
# all_y = []
# xx = []
# yy = []
# for i in range(len(train_set)):
#     x_batch, y_batch = train_set[i]
#     x_b, y_b = test_set[i]
#     all_x.append(x_batch)
#     all_y.append(y_batch)
#     xx.append(x_b)
#     yy.append(y_b)
    

# x = np.concatenate(all_x, axis=0)
# y = np.concatenate(all_y, axis=0)
# xx = np.concatenate(xx, axis=0)
# yy = np.concatenate(yy, axis=0)
# # print('x.shape: ', x.shape) # x.shape:  (160, 100, 100, 3)
# # print('y.shape: ', y.shape) # y.shape:  (160,)

    ### save data as npy
np.save(D_path + 'k46_horse_x_tr.npy', arr = train[0][0])
np.save(D_path + 'k46_horse_y_tr.npy', arr = train[0][1])
np.save(D_path + 'k46_horse_x_ts.npy', arr = test[0][0])
np.save(D_path + 'k46_horse_y_ts.npy', arr = test[0][1])

###### model
    ### modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(32,(8,8), input_shape = (100,100,3), activation = 'relu'))
model.add(Conv2D(26,(8,8), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(20,(8,8), activation = 'relu'))
model.add(Conv2D(14,(16,16), activation = 'relu'))
model.add(MaxPool2D())
model.add(Conv2D(12,(16,16), activation = 'relu'))
model.add(Conv2D(10,(8,8), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(8,(4,4), activation = 'relu'))
model.add(Flatten())
model.add(Dense(28))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

###### compile / fit
    ### compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    ### fit
hist = model.fit(train[0][0],train[0][1], epochs = 20, batch_size = 1, 
                validation_split = 0.1, callbacks = [], verbose=2)

###### evaluate / predict
from sklearn.metrics import r2_score

loss = model.evaluate(test[0][0], test[0][1])
pred = model.predict(test[0][0])
r2 = r2_score(test[0][1], pred)

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


