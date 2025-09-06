import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

import random
random_seed = 55
tf.random.set_seed(random_seed)

# 1. 데이터
datasets = load_iris()

x = datasets.data   # (150, 4)
y = datasets.target # (150, 4)

values, counts = np.unique(y, return_counts=True)
for value, count in zip(values, counts):
    print(f"{value} : {count}")
'''
0 : 50
1 : 50
2 : 50
각각 순서대로 0부터 2까지의 숫자로 대치했을 뿐인데 수치 상의 거리라는 의미가 생겼다.
'''

# OneHotEncoding
# 1. sklearn
from sklearn.preprocessing import OneHotEncoder
# Default는 sparse=True
onehot = OneHotEncoder(sparse=False) # sparse=True일 때는 scipy의 sparse matrix로 저장, toarray()로 변환 필요
y_encoded = onehot.fit_transform(y.reshape(-1, 1))
'''
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.] 
    ...
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]
'''

# 2. pandas
y_dummies = pd.get_dummies(y, drop_first=False, prefix='target')
'''
     target_0  target_1  target_2
0           1         0         0
1           1         0         0
2           1         0         0
3           1         0         0
4           1         0         0
..        ...       ...       ...
145         0         0         1
146         0         0         1
147         0         0         1
148         0         0         1
149         0         0         1
'''

# 3. keras
from tensorflow.keras.utils import to_categorical
y_categorical = to_categorical(y, num_classes=3)
'''
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
    ...
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]]
'''

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y_encoded,
    stratify=y,
    test_size=0.1,
    random_state=random_seed
)

# 2. 모델 구성
model = Sequential()
model.add(Input(shape=(4, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3. 컴파일, 훈련
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    patience=50,
    restore_best_weights=True
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.1,
    verbose=2,
    callbacks=[es]
)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=0)
print(f"loss : {loss[0]}")
print(f"acc : {loss[1]}")
