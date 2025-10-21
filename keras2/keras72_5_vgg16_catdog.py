import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D
import tensorflow as tf
import random

SEED = 333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet152, ResNet152V2, DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2

vgg16 = VGG16(
    include_top=False,
    input_shape=(32, 32, 3),   
)

vgg16.trainable = True

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

# 실습
# 비교 할 데이터
# 1. 이전 내가 한 최상의 결과가
# 2. 가중치 동결 X, trainable = True
# 3. 가중치 동결 O, trainable = False
# 시간까지 비교
# Flatten 과 GAP

####################################
# cifar10
# cifar100
# horse
# rps
# kaggle cat dog
# man women
####################################