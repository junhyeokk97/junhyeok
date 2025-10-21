from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,BatchNormalization, Input
import time
import numpy as np
import matplotlib as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler

D_path = './_data/kaggle/tensor_cert/rps/'

    ### load saved data
x_train = np.load(D_path + 'k46_rps_x_tr.npy')
y_train = np.load(D_path + 'k46_rps_y_tr.npy')
x_test = np.load(D_path + 'k46_rps_x_ts.npy')
y_test = np.load(D_path + 'k46_rps_y_ts.npy')

print(x_train.shape, y_train.shape) # (2048, 100, 100, 3) (2048, 3)

augment_size=1000

aaa = np.tile(x_train[0].reshape(100*100*3), augment_size).reshape(-1, 100, 100, 3)
print(aaa.shape)    # (1000, 100, 100, 3)

datagen = ImageDataGenerator(
    rescale=1./255,
)

randidx = np.random.randint(x_train.shape[0], size=augment_size)
# print(randidx)
print(np.min(randidx), np.max(randidx)) # 3 2042

x_augment = x_train[randidx].copy()
y_augment = y_train[randidx].copy()

print(x_augment)
print(x_augment.shape)  # (1000, 100, 100, 3)
print(y_augment.shape)  # (1000, 3)

x_augment = x_augment.reshape(
    x_augment.shape[0],
    x_augment.shape[1],
    x_augment.shape[2],3
)
print(x_augment.shape) # (1000, 100, 100, 3)

x_augment = datagen.flow(
    x_augment,
    y_augment,
    batch_size=augment_size,
    shuffle=True
).next()[0]

x_train = np.concatenate((x_train, x_augment))
y_train = np.concatenate((y_train, y_augment))
print(x_train.shape, y_train.shape) # (3048, 100, 100, 3) (3048, 3)

model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, input_shape=(100,100,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=(2,2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Conv2D(8, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=4))
model.add(Dense(units=3, activation='softmax'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', mode='max', patience=20, verbose=2, restore_best_weights=True)


hist = model.fit(x_train, y_train, epochs=1000, verbose=2, batch_size=200,
                 validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # evaluate에도 verbose 사용 가능.
print('loss: ', loss[0])
print('acc: ', loss[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=0)


# print('time: ', end-start)
import matplotlib.pyplot as plt
plt.imshow(x_test[1])
plt.show()