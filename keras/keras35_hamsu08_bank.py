from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import numpy as np
import pandas as pd
import tensorflow as tf

# 1. 데이터
path = './_data/kaggle/bank/'
train_df = pd.read_csv(path+'train.csv', index_col=0)
test_df = pd.read_csv(path+'test.csv', index_col=0)
submission_df = pd.read_csv(path+'sample_submission.csv')

le = LabelEncoder()
train_df['Geography'] = le.fit_transform(train_df['Geography'])
test_df['Geography'] = le.transform(test_df['Geography'])
train_df['Gender'] = le.fit_transform(train_df['Gender'])
test_df['Gender'] = le.transform(test_df['Gender'])

train_df = train_df.drop(columns=['CustomerId','Surname'], axis=1)
test_df = test_df.drop(columns=['CustomerId','Surname'], axis=1)

x = train_df.drop(columns=['Exited'], axis=1)
y = train_df['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.1,
    random_state=100
)

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# 2. 모델 구성 10
input = Input(shape=(10,))
dense1 = Dense(10)(input)
drop1 = Dense(10)(dense1)
dense2 = Dense(10)(drop1)
dense3 = Dense(10)(dense2)
drop2 = Dense(10)(dense3)
dense4 = Dense(10)(drop2)
output = Dense(1)(dense4)
model = Model(inputs= input, outputs= output)
model.summary()
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 10)]              0
#  dense (Dense)               (None, 10)                110
#  dense_1 (Dense)             (None, 10)                110
#  dense_2 (Dense)             (None, 10)                110
#  dense_3 (Dense)             (None, 10)                110
#  dense_4 (Dense)             (None, 10)                110
#  dense_5 (Dense)             (None, 10)                110
#  dense_6 (Dense)             (None, 1)                 11
# =================================================================
# Total params: 671
# Trainable params: 671
# Non-trainable params: 0

# 3. 컴파일, 훈련
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=30,
    restore_best_weights=True
)

import time
start = time.time()

model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=100,
    class_weight=class_weights,
    verbose=2
)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
loss = results[0]
acc = results[1]
print("Loss:", loss)
print("Accuracy :", acc)

import tensorflow as tf

end = time.time()

print("걸린시간: ", end - start)

gpus = tf.config.list_physical_devices('GPU')           # tensorflow 2.7.4 / 2.9.0 = cpu버전

if gpus:
    print('GPU 있다.')
else:
    print('GPU 없다.')
    
# Loss: 0.688507616519928
# Accuracy : 0.7874454855918884
# 걸린시간:  57.25161099433899
# GPU 있다.

# Loss: 0.6950567960739136
# Accuracy : 0.21255452930927277
# 걸린시간:  20.40525531768799
# GPU 없다.