import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, MaxPooling2D
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

train_data = ImageDataGenerator(
    rescale=1./255,
)

test_data = ImageDataGenerator(
    rescale=1./255,
)

train_path = './_data/kaggle/cat_dog/train2/'
test_path = './_data/kaggle/cat_dog/test2/'
sub_path = './_data/kaggle/cat_dog/'
sub_csv = pd.read_csv(sub_path + 'sample_submission.csv')

xy_train = train_data.flow_from_directory(
    train_path,
    target_size=(200, 200),
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

xy_test = test_data.flow_from_directory(
    test_path,
    target_size=(200, 200),
    batch_size=25000,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

print(xy_train[0][0].shape) # (100, 200, 200, 3)
print(xy_train[0][1].shape) # (100,)
print(len(xy_train))        # 250
all_x = []
all_y = []

# for i in range(len(xy_train)):
#     x_batch, y_batch = xy_train[i]
#     all_x.append(x_batch)
#     all_y.append(y_batch)
# # print(all_x)

# ### 리스트를 하나의 numpy 배열로 합친다. (사슬처럼 엮는다.)
# x = np.concatenate(all_x, axis=0)
# y = np.concatenate(all_y, axis=0)
# print('x.shape: ', x.shape) # x.shape:  (25000, 200, 200, 3)
# print('y.shape: ', y.shape) # y.shape:  (25000,)

x_train = xy_train[0][0]      # 통배치용
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape, y_train.shape) # (10, 150, 150, 3) (10,)
print(x_test.shape, y_test.shape)   # (10, 150, 150, 3) (10,)

model = Sequential()
model.add(Conv2D(128, (3,3), strides=1, input_shape=(200,200,3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=8))
model.add(Dense(units=1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', patience=30,
                   restore_best_weights=True)


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

start = time.time()
model.fit(x_train, y_train, epochs=100000, validation_split=0.2,
          verbose=2, callbacks=[es])
end = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)

print('loss: ', loss[0])
print('acc: ', loss[1])
print('time: ', end-start)

# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
# submission_csv['target']
# submission_csv.to_csv(path + 'submission_1.csv')

import matplotlib.pyplot as plt
plt.imshow(x_train[1], 'gray')
plt.show()

