from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,BatchNormalization, Input
import time
import numpy as np
import matplotlib as plt
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train/255.
# x_test = x_test/255.
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


print(x_train.shape) # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)

augment_size = 40000

aaa =np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1 , 28, 28, 1)
print(aaa)

datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,    
    vertical_flip=True,)

randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augment = x_train[randidx].copy()
y_augment = y_train[randidx].copy()

print(x_augment)
print(x_augment.shape)  # (40000, 28, 28)
print(y_augment.shape)  # (40000,)

x_augment = x_augment.reshape(
    x_augment.shape[0],
    x_augment.shape[1],
    x_augment.shape[2],1
)

x_augment = datagen.flow(
    x_augment,
    y_augment,
    batch_size = augment_size,
    shuffle = False,
    save_to_dir='c:/study25/_data/_save_img/02_mnist/'
).next()[0]

print(x_augment.shape)  # (40000, 28, 28, 1)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augment))
y_train = np.concatenate((y_train, y_augment))
print(x_train.shape, x_test.shape)  # (100000, 28, 28, 1) (10000, 28, 28, 1)

model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, input_shape=(28, 28, 1)))   # 27,27,64
model.add(Conv2D(filters=32, kernel_size=(3,3)))        # 25,25,32
model.add(Conv2D(16, (3,3)))                            # 23,23,16
model.add(Flatten())
model.add(Dense(units=16))
model.add(Dense(units=16))
model.add(Dense(units=1, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=2, restore_best_weights=True)


hist = model.fit(x_train, y_train, epochs=1000, verbose=2,
                 validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # evaluate에도 verbose 사용 가능.
print('loss: ', loss[0])
print('acc: ', loss[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('accuracy_score: ', acc)


import matplotlib.pyplot as plt
plt.imshow(x_train[1], 'gray')
plt.show()