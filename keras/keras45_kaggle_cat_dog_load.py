import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import time

train_data = ImageDataGenerator(
    rescale=1./255,
)

test_data = ImageDataGenerator(
    rescale=1./255,
)


np_path = 'c:/study25/_data/_save_npy/'
x = np.load(np_path + "keras44_02_x_train.npy")
y = np.load(np_path + "keras44_02_y_train.npy")
test = np.load(np_path + "keras44_02_test.npy")

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50)

model = Sequential()
model.add(Conv2D(128, (3,3), strides=1, input_shape=(100,100,1)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=8))
model.add(Dense(units=1, activation='sigmoid'))


path = './_data/kaggle/cat_dog/ff/'
filename = 'cat_dog_{epoch:04d}_{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath=path+filename
)



model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', patience=19,
                   restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=10000, validation_split=0.2,
          verbose=2, callbacks=[es])
end = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print('loss: ', loss[0])
print('acc: ', loss[1])
print('time: ', end-start)


# sub_path = './_data/kaggle/cat_dog/'
# sub_csv = pd.read_csv(sub_path + 'sample_submission.csv')
# y_submit = model.predict(x_test)
# sub_csv['label'] = y_submit
# sub_csv.to_csv(sub_path + 'submission_1.csv', index=False)

import matplotlib.pyplot as plt
plt.imshow(x_train[1], 'gray')
plt.show()