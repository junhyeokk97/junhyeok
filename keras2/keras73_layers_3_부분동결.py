import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D
import tensorflow as tf
import random
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.metrics import accuracy_score

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

augment_size = 50000

datagen = ImageDataGenerator(
    rescale=1./255, 
    # horizontal_flip=True,    
    # vertical_flip=True,      
    # width_shift_range=0.3,   
    # height_shift_range=0.23,  
    # rotation_range=10,        
    # zoom_range=1.2,          
    # shear_range=0.7,         
    # fill_mode='nearest'
)

vgg16 = VGG16(
    include_top=False,
    input_shape=(32, 32, 3),   
)

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.summary()

print(len(model.weights))
print(len(model.trainable_weights))

# vgg16.trainable = True
# print(len(model.weights))            32
# print(len(model.trainable_weights))  32

# vgg16.trainable = False
# print(len(model.weights))            32
# print(len(model.trainable_weights))  6

# 1. 전체동결1

# 2. 전체동결2
for layer in model.layers:
    layer.trainable = False

# 3. 부분동결
# model.layers[2].trainable = False

import pandas as pd
pd.set_option('max_colwidth', None) # None 길이 , 10개
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer type', 'Layer Name', 'Layer Trainable'])
print(results)