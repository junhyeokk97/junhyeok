import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

###### preparing data
    ### load data from npy
import time

D_path = 'c:/before/study25/_data/kaggle/cat_dog/'
x_tr = np.load(D_path + 'k45_x_tr.npy')
y_tr = np.load(D_path + 'k45_y_tr.npy')
x_ts = np.load(D_path + 'k45_x_ts.npy')
y_ts = np.load(D_path + 'k46_y_ts.npy')

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

print(x_tr.shape, y_tr.shape, x_ts.shape, y_ts.shape)
# (1000, 200, 200, 3) (1000,) (500, 200, 200, 3) (500,)

###### model
    ### load model
    
model = load_model(D_path + 'model.hdf5')

me = np.load('./_data/image/me/keras47_me.npy')

loss = model.evaluate(x_ts, y_ts)
pred = model.predict(me)

from sklearn.metrics  import r2_score

print('loss: ', loss[0])
print('acc: ', loss[1])
if pred == 0:
    print('cat!!')
else:
    print('dog!!')

