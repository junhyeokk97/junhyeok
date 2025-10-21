from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,BatchNormalization, Input
import time
import numpy as np
import matplotlib as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler

np_path = 'c:/study25/_data/_save_npy/'
x = np.load(np_path + "keras44_02_x_train.npy")
y = np.load(np_path + "keras44_02_y_train.npy")
test = np.load(np_path + "keras44_02_test.npy")
start = time.time()
print(x.shape, y.shape, test.shape)
# (25000, 100, 100, 1) (25000,) (12500, 100, 100, 1)
augment_size = 5000
aaa = np.tile(x[0].reshape(100*100), augment_size).reshape(-1, 100, 100, 1)
print(aaa.shape)    # (5000, 100, 100, 1)

datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,    
    vertical_flip=True,
)

randidx = np.random.randint(x.shape[0], size=augment_size)
print(randidx)
print(np.min(randidx), np.max(randidx))

x_augment = x[randidx].copy()
y_augment = y[randidx].copy()

print(x_augment)
print(x_augment.shape)    # (5000, 100, 100, 1)
print(y_augment.shape)    # (5000,)

x_augment = x_augment.reshape(
    x_augment.shape[0],
    x_augment.shape[1],
    x_augment.shape[2],1
)
print(x_augment.shape)  # (5000, 100, 100, 1)

x_augment = datagen.flow(
    x_augment,
    y_augment,
    batch_size=augment_size,
    shuffle=True,
    save_to_dir='c:/study25/_data/_save_img/05_cat_dog/'
).next()[0]

print(x_augment.shape)    # (5000, 100, 100, 1)
print(x.shape)            # (25000, 100, 100, 1)

x = np.concatenate((x, x_augment))
y = np.concatenate((y, y_augment))
print(x.shape, y.shape) # (30000, 100, 100, 1) (30000,)
end = time.time()

model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, input_shape=(100,100,1)))
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
model.add(Dense(units=1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', mode='max', patience=20, verbose=2, restore_best_weights=True)


hist = model.fit(x, y, epochs=1000, verbose=2, batch_size=400,
                 validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(test, verbose=1)    # evaluate에도 verbose 사용 가능.
print('loss: ', loss[0])
print('acc: ', loss[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(test)
y_predict = np.argmax(y_predict, axis=0)
acc = accuracy_score(test, y_predict)

print('accuracy_score: ', acc)
print('time: ', end-start)
import matplotlib.pyplot as plt
plt.imshow(x[1])
plt.show()