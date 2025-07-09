from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import time
# 1. 데이터
path = "./_data/dacon/diabetes/"
train_df = pd.read_csv(path+'train.csv', index_col=0)
test_df = pd.read_csv(path+'test.csv', index_col=0)
submission_df = pd.read_csv(path+'sample_submission.csv')

x = train_df.drop(columns=['Outcome'], axis=1)
y = train_df['Outcome']

x[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = x[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)
x = x.fillna(x.mean())

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    stratify=y,
    test_size=0.1,
    random_state=100,
)

# 2. 모델 구성
model = Sequential([
    Dense(64, input_shape=(x_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 3. 컴파일, 훈련
# early = EarlyStopping(
#     monitor='val_loss',
#     mode='auto',
#     patience=30,
#     restore_best_weights=True,
# )

# path = './_save/keras28_mcp/07_dacon_diabetes/'
# filename = '07_dacon_diabetes_{epoch:04d}_{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     save_best_only=True,
#     save_weights_only=False,
#     filepath=path+filename
# )

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)
start = time.time()
model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=32,
    verbose=2,
    validation_split=0.2,
)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
loss = results[0]
acc = results[1]
print("Loss:", loss)
print("Accuracy :", acc)

'''
Loss: 0.622363805770874
Accuracy : 0.6515151262283325
'''

import tensorflow as tf

end = time.time()

print("걸린시간: ", end - start)

gpus = tf.config.list_physical_devices('GPU')           # tensorflow 2.7.4 / 2.9.0 = cpu버전

if gpus:
    print('GPU 있다.')
else:
    print('GPU 없다.')
    
# Loss: 0.5594774484634399
# Accuracy : 0.7272727489471436
# 걸린시간:  9.776158571243286
# GPU 있다.

# Loss: 0.5676708221435547
# Accuracy : 0.7121211886405945
# 걸린시간:  3.9826738834381104
# GPU 없다.