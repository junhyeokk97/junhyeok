import numpy as np
import pandas as pd

###### preparing data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

D_path = './_data/kaggle/tensor_cert/rps/'

    ### load saved data
x_tr = np.load(D_path + 'k46_rps_x_tr.npy')
y_tr = np.load(D_path + 'k46_rps_y_tr.npy')
x_ts = np.load(D_path + 'k46_rps_x_ts.npy')
y_ts = np.load(D_path + 'k46_rps_y_ts.npy')

###### model
    ### modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(8,(8,8), input_shape = (100,100,3), activation = 'relu'))
model.add(Conv2D(16,(8,8)))
model.add(Conv2D(32,(8,8)))
model.add(MaxPool2D())
model.add(Conv2D(64,(16,16)))
model.add(Conv2D(32,(8,8)))
model.add(MaxPool2D())
model.add(Conv2D(8,(4,4)))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4))
model.add(Dense(3, activation = 'softmax'))
model.summary()

###### compile / fit
    ### compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

    ### fit
hist = model.fit(x_tr, y_tr, epochs = 20, batch_size = 16, 
                validation_split = 0.1, callbacks = [])

###### evaluate / predict
from sklearn.metrics import r2_score

loss = model.evaluate(x_ts,y_ts)
pred = model.predict(x_ts)
r2 = r2_score(np.argmax(y_ts, axis = 1), np.argmax(pred, axis = 1))
print('loss : ', loss[0])
print('acc : ', loss[1])
print('r2 : ', r2)


