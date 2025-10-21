from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D,BatchNormalization, Input
import time
import numpy as np
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(x_train[0].shape) # (28, 28)
start = time.time()
augment_size = 40000  # 데이터 6만개를 10만개로 만든다.

 
aaa = np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28,28,1) # augment # of data to 'augment_size' times
print(aaa.shape)    # (40000, 28, 28, 1)  > agment_size 만큼 x_train 1번째 이미지를 만들어낸다.

datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,    
    # vertical_flip=True,      
    # width_shift_range=0.3,   
    # height_shift_range=0.23,  
    # rotation_range=10,        
    # zoom_range=1.2,          
    # shear_range=0.7,         
    # fill_mode='nearest'
)


randidx = np.random.randint(x_train.shape[0], size=augment_size)
        # np.random.randint(60000, 40000)
print(randidx)  # [34261 40902 15794 ... 25353   965 35325]
print(np.min(randidx), np.max(randidx)) # 0 59999

x_augmented = x_train[randidx].copy()   # 4만개의 데이터 copy / copy로 새로운 메모리 할당.
                                        # 서로 영향 없음.
y_augmented = y_train[randidx].copy()

print(x_augmented) 
print(x_augmented.shape)    # (40000, 28, 28)
print(y_augmented.shape)    # (40000,)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2],1
)   
print(x_augmented.shape)    # (40000, 28, 28, 1)

x_augmented = datagen.flow(
    x_augmented,    # x datas
    y_augmented, # y datas which is all zero // y데이터 생성, 전부 0으로 된 y값.
    batch_size = augment_size,
    shuffle = False,
    save_to_dir='c:/study25/_data/_save_img/01_fashion/'    # dir 저장
    ).next()[0]     # x_augment를 뺄거면 [0]번째

print(x_augmented.shape)    # (40000, 28, 28, 1)
print(x_train.shape)        # (60000, 28, 28)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))
print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)
y_augmented = ohe.fit_transform(y_augmented)

end = time.time()
model = Sequential()
model.add(Conv2D(256, (3,3), strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=128, kernel_size=(2,2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=16))
model.add(Dense(units=1, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_acc', mode='max', patience=35, verbose=2, restore_best_weights=True)


hist = model.fit(x_train, y_train, epochs=1000, verbose=2, batch_size=100,
                 validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # evaluate에도 verbose 사용 가능.
print('loss: ', loss[0])
print('acc: ', loss[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=0)

print('time: ', end-start)
import matplotlib.pyplot as plt
plt.imshow(x_train[1])
plt.show()