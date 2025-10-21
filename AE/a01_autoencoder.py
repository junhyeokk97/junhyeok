import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input

#1. 데이터
(x_trian, _), (x_test, _) = mnist.load_data()

x_train = x_trian.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. 모델
input_img = Input(shape=(28*28,))

#### 인코더
# encoded = Dense(1, activation='relu')(input_img)      # 너무 많이 없어짐
encoded = Dense(64, activation='relu')(input_img)   
# encoded = Dense(1024, activation='relu')(input_img)   # 거의 그대로 남음  

#### 디코더
# decoded = Dense(28*28, activation='linear')(encoded)
# decoded = Dense(28*28, activation='relu')(encoded)
decoded = Dense(28*28, activation='sigmoid')(encoded)   # /255.으로 0~1사이 정규화 했기때문에 가능
# decoded = Dense(28*28, activation='tanh')(encoded)    # 이건 별로임

autoencoder = Model(input_img, decoded)

#3. 컴파일, 훈련
# autoencoder.compile(loss='mse', optimizer='adam')
autoencoder.compile(loss='binary_crossentropy', optimizer='adam')   # 0과 1이기 때문에 가능
autoencoder.fit(x_train, x_train, epochs=50, batch_size=32, validation_split=0.2)

#4. 평가, 예측
decoded_img = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(15, 4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()