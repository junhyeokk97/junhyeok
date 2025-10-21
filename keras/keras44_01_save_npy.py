import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

train_data = ImageDataGenerator(
    rescale=1./255,
)

test_data = ImageDataGenerator(
    rescale=1./255,
)

train_path = './_data/kaggle/cat_dog/train2/'
test_path = './_data/kaggle/cat_dog/test2/'


xy_train = train_data.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)

xy_test = test_data.flow_from_directory(
    test_path,
    target_size=(100, 100),
    batch_size=500,
    class_mode='binary',
    color_mode='grayscale',
)

print(xy_train[0][0].shape) # (500, 100, 100, 1)
print(xy_train[0][1].shape) # (500,)
print(len(xy_train))        # 50
all_x = []
all_y = []
all_test = []

for i in range(len(xy_train)):
    x_batch, y_batch = xy_train[i]
    all_x.append(x_batch)
    all_y.append(y_batch)
    
for i in range(len(xy_test)):
    x_batch, y_batch = xy_test[i]
    all_test.append(x_batch)


### 리스트를 하나의 numpy 배열로 합친다. (사슬처럼 엮는다.)
x = np.concatenate(all_x, axis=0)
y = np.concatenate(all_y, axis=0)
test = np.concatenate(all_test, axis=0)

np_path = 'c:/study25/_data/_save_npy/'
np.save(np_path + "keras44_02_x_train.npy", arr=x)
np.save(np_path + "keras44_02_y_train.npy", arr=y)
np.save(np_path + "keras44_02_test.npy", arr=test)


