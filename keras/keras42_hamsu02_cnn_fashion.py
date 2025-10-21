import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Flatten, MaxPooling2D, Input
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 28 ,28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)



# model = Sequential()
# model.add(Conv2D(64, (3,3), strides=1, input_shape=(28,28,1)))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=64, kernel_size=(3,3)))
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Conv2D(32, (2,2), activation='relu'))
# model.add(Conv2D(16, (2,2), activation='relu'))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dense(units=16))
# model.add(Dense(units=10, activation='softmax'))
# model.summary()

input = Input(shape=(28,28,1))
conv1 = Conv2D(64, (3,3))(input)
max1 = MaxPooling2D()(conv1)
conv2 = Conv2D(64, (3,3))(max1)
max2 = MaxPooling2D()(conv2)
drop1 = Dropout(0.2)(max2)
bn1 = BatchNormalization()(drop1)
conv3 = Conv2D(32, (2,2), activation='relu')(drop1)
conv4 = Conv2D(16, (2,2), activation='relu')(conv3)
drop2 = Dropout(0.2)(conv4)
bn2 = BatchNormalization()(drop2)
flat = Flatten()(bn2)
dense1 = Dense(16)(flat)
output = Dense(10, activation='softmax')
model = Model(inputs=input, outputs=output)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=2, restore_best_weights=True)


hist = model.fit(x_train, y_train, epochs=1000, batch_size=200, verbose=2,
                 validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)    # evaluate에도 verbose 사용 가능.
print('loss: ', loss[0])
print('acc: ', loss[1])

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test.to_numpy(), axis=1)
acc = accuracy_score(y_test, y_predict)

print('accuracy_score: ', acc)

# loss:  0.2838459014892578
# acc:  0.897599995136261
# accuracy_score:  0.8976

# loss:  0.277344673871994
# acc:  0.9000999927520752
# accuracy_score:  0.9001

# loss:  0.27829915285110474
# acc:  0.9021000266075134
# accuracy_score:  0.9021