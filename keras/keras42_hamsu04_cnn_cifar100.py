from tensorflow.keras.datasets import cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D ,Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

y_train = pd.get_dummies(y_train.reshape(-1))
y_test = pd.get_dummies(y_test.reshape(-1))
print(y_train.shape, y_test.shape)   # (50000, 100) (10000, 100)

# model = Sequential()
# model.add(Conv2D(100, (2,2), strides=1, input_shape=(32,32,3)))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=200, kernel_size=(2,2)))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Conv2D(170, (3,3), activation='relu'))
# model.add(Conv2D(150, (3,3), activation='relu'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dense(units=110))
# model.add(Dense(units=100, activation='softmax'))
# model.summary()

input = Input(shape=(32,32,3))
conv1 = Conv2D(100, (2,2))(input)
max1 = MaxPooling2D()(conv1)
conv2 = Conv2D(200, (3,3))(max1)
max2 = MaxPooling2D()(conv2)
drop1 = Dropout(0.2)(max2)
bn1 = BatchNormalization()(drop1)
conv3 = Conv2D(170, (3,3), activation='relu')(drop1)
conv4 = Conv2D(150, (3,3), activation='relu')(conv3)
drop2 = Dropout(0.2)(conv4)
bn2 = BatchNormalization()(drop2)
flat = Flatten()(bn2)
dense1 = Dense(110)(flat)
output = Dense(100, activation='softmax')
model = Model(inputs=input, outputs=output)
model.summary() 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=17,
                   restore_best_weights=True)
# y_train_label = np.argmax(y_train.to_numpy(), axis=1)
# class_weight = compute_class_weight(class_weight='balanced',
#                           classes=np.unique(y_train),
#                           y=y_train)
# cw = dict(enumerate(class_weight))

model.fit(x_train, y_train, epochs=1000, batch_size=500, verbose=2,
          validation_split=0.2, callbacks=[es])

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test.to_numpy(), axis=1)
acc = accuracy_score(y_test, y_predict)

print('loss: ', loss[0])
print('acc: ', loss[1])

import matplotlib.pyplot as plt
plt.imshow(x_train, 'gray')
plt.show()


# loss:  2.4642913341522217
# acc:  0.42719998955726624

# loss:  2.7869577407836914
# acc:  0.4300000071525574