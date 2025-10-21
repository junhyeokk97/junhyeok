from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, MaxPooling2D,BatchNormalization, Input
import time
import numpy as np
import matplotlib as plt
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = x_train/255.
# x_test = x_test/255.

print(x_train.shape)    # (50000, 32, 32, 3)
print(x_train[0].shape) # (32, 32, 3)

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

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx)
print(np.min(randidx), np.max(randidx))

x_augment = x_train[randidx].copy()
y_augment = y_train[randidx].copy()
# y_augment = y_augment.reshape(-1,)

print(x_augment) 
print(x_augment.shape)    # (50000, 32, 32, 3)
print(y_augment.shape)    # (50000, 1)

x_augmented = datagen.flow(
    x_augment,    # x datas
    y_augment, # y datas which is all zero // y데이터 생성, 전부 0으로 된 y값.
    batch_size = augment_size,
    shuffle = False,
    # save_to_dir='c:/study25/_data/_save_img/03_cifar10/' 
    ).next()[0]

print(x_augmented.shape)    # (50000, 32, 32, 3)
print(x_train.shape)        # (50000, 32, 32, 3)
print(y_train.shape)        # (50000, 32, 32, 3)

x_train = np.concatenate((x_train, x_augment))
y_train = np.concatenate((y_train, y_augment))
print(x_train.shape, y_train.shape)     # (100000, 32, 32, 3) (100000, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2], 3)
y_train = y_train.reshape(y_train.shape[0], 1)

print(x_train.shape)

model = Sequential()
model.add(LSTM(20, input_shape=(1024, 3)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(units=16))
model.add(Dense(units=8))
model.add(Dense(units=4))
model.add(Dense(units=1, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=2,
                   restore_best_weights=True)
str = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=2,
          validation_split=0.2, callbacks=[es])
end = time.time()


loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('loss: ', loss[0])
print('acc: ', loss[1])
print('걸린시간: ', end-str)
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0])
# plt.show()