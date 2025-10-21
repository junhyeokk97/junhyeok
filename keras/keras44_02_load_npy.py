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

np_path = 'c:/study25/_data/_save_npy'
# np.save(np_path + "keras44_01_x_train.npy", arr=x)
# np.save(np_path + "keras44_01_y_train.npy", arr=y)
x = np.load(np_path + "keras44_01_x_train.npy")
y = np.load(np_path + "keras44_01_y_train.npy")
test = np.load(np_path + "keras44_01_x_test.npy")



print(x)
print(y[:20])
print(x.shape, y.shape)

