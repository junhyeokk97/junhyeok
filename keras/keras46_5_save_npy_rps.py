import numpy as np
import pandas as pd

###### preparing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

    ### image to numpy
D_path = './_data/tensor_cert/rps/'
train_set = ImageDataGenerator(rescale = 1./255,
                               horizontal_flip = True,
                               width_shift_range = 0.3,
                               vertical_flip = True,
                               height_shift_range = 0.4)
test_set = ImageDataGenerator(rescale = 1./255)

train = train_set.flow_from_directory(D_path,
                                    target_size = (100,100),
                                    batch_size = 100,
                                    class_mode = 'categorical',
                                    color_mode = 'rgb',
                                    shuffle = True,
                                    seed = 50)
test = test_set.flow_from_directory(D_path,
                                    target_size = (100,100),
                                    batch_size = 100,
                                    class_mode = 'categorical',
                                    color_mode = 'rgb',
                                    shuffle = True,
                                    seed = 50)

print(train[0][0].shape, test[0][0].shape) # (100, 100, 100, 3) (100, 100, 100, 3)
print(train[0][1].shape, test[0][1].shape) # (100, 3) (100, 3)
print(len(train), len(test)) # 21 21

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
print('x.shape: ', x.shape) # x.shape:  (2048, 100, 100, 3)
print('y.shape: ', y.shape) # y.shape:  (2048, 3)

    ### save data as npy
np.save(D_path + 'k46_rps_x_tr.npy', arr = x)
np.save(D_path + 'k46_rps_y_tr.npy', arr = y)
np.save(D_path + 'k46_rps_x_ts.npy', arr = xx)
np.save(D_path + 'k46_rps_y_ts.npy', arr = yy)

###### model
    ### modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(64,(3,3), input_shape = (100,100,3), activation = 'relu'))
model.add(Conv2D(32,(3,3), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16,(2,2), activation = 'relu'))
model.add(MaxPool2D())
model.add(Conv2D(8,(3,3), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Conv2D(4,(3,3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(28))
model.add(Dropout(0.2))
model.add(Dense(14))
model.add(Dense(7))
model.add(Dense(3, activation = 'softmax'))
model.summary()

###### compile / fit
    ### compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

    ### fit
hist = model.fit(x,y, epochs = 10, batch_size = 32, 
                validation_split = 0.1, callbacks = [])

###### evaluate / predict
from sklearn.metrics import accuracy_score

loss = model.evaluate(xx, yy)
pred = model.predict(xx)
print('loss : ', loss[0])
print('acc : ', loss[1])
