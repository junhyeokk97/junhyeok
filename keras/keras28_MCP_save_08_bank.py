from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

# 2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

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

path = './_save/keras28_mcp/08_bike/'
filename = '08_bike_{epoch:04d}_{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_weights_only=False,
    filepath=path+filename
)

model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=1000,
    batch_size=128,
    callbacks=[es, mcp],
    class_weight=class_weights,
    verbose=2
)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
loss = results[0]
acc = results[1]
print("Loss:", loss)
print("Accuracy :", acc)