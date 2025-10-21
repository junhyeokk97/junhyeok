# 인코더(28, 28, 1)
# Conv2D (28, 28, 1)   padding = same
# maxpool (14, 14, 64)
# conv (14, 14, 32) padding = same
# maxpool(7, 7, 32)

# 디코더
# conv(7, 7, 32)  padding = smae
# upsmapling2D(2, 2) (12, 12 ,32)
# conv (14, 14, 1)  padding = same
# upsampling2D(2, 2) (28, 28, 16)
# conv(28, 28, 1)

# conv와 maxpooling 으로 노드를 줄이고   
# upsmapling으로 다시 늘린다 ( 모래시계 형태 )

# (28, 28, 1)
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, UpSampling2D, MaxPooling2D
from sklearn.decomposition import PCA

#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
                                 # (평균, 표준편차 0.1인 정규분포형태의 랜덤값, size)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

print(x_train_noised.shape, x_test_noised.shape)        # (60000, 784) (10000, 784)
print(np.max(x_train), np.min(x_test))                  # 1.0 0.0
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.4611409472854433 -0.5246125806788717

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
print(np.max(x_train_noised), np.min(x_test_noised))    # 1.0 0.0

#2. 모델
input_img = Input(shape=(28*28,))

hidden_size = [1238, 64, 31, 64, 128]
hidden_size = [64, 128, 256, 128, 64]
hidden_size = [128, 128, 128, 128, 128]

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(hidden_layer_size, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    return model

# 인코더(28, 28, 1)
# Conv2D (28, 28, 1)   padding = same
# maxpool (14, 14, 64)
# conv (14, 14, 32) padding = same
# maxpool(7, 7, 32)

# 디코더
# conv(7, 7, 32)  padding = smae
# upsmapling2D(2, 2) (12, 12 ,32)
# conv (14, 14, 1)  padding = same
# upsampling2D(2, 2) (28, 28, 16)
# conv(28, 28, 1)

# conv와 maxpooling 으로 노드를 줄이고   
# upsmapling으로 다시 늘린다 ( 모래시계 형태 )

# 0.95 이상  : 
# 0.99 이상  : 
# 0.999 이상 : 
# 1.0 이상   : 

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=8)
model_03 = autoencoder(hidden_layer_size=32)
model_04 = autoencoder(hidden_layer_size=64)
model_05 = autoencoder(hidden_layer_size=154)
model_06 = autoencoder(hidden_layer_size=331)
model_07 = autoencoder(hidden_layer_size=486)
model_08 = autoencoder(hidden_layer_size=713)


# model_list = [1,8,32,64,154,331,486,713]
# for i, hidden in enumerate(model_list, 1):
#     model = autoencoder(model_list = model_list)
#     model.compile(loss='binary_crossentropy', optimizer='adam')
#     model.fit(x_train_noised, x_train_noised, epochs=50, batch_size=128, validation_split=0.2, verbose=0)
#     model_list.append(model)   

# 3. 컴파일, 훈련 / 4. 평가, 예측
hidden_layer_size=[1,8,32,64,154,331,486,713]
outputs = []
outputs.append(x_test)

for i in hidden_layer_size:
    print('{i}개')
    model = autoencoder(hidden_layer_size=i)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x_train_noised, x_train, epochs=50, batch_size=128, validation_split=0.2, verbose=0)

    decoded_imgs = model.predict(x_test_noised)
    outputs.append(decoded_imgs)

import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(9, 5, figsize=(15,15))

random_images = random.sample(range(decoded_imgs.shape[0]), 5)

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()