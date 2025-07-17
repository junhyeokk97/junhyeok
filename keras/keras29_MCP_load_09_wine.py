import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(datasets.feature_names)

print(x.shape)  # (178, 13)
print(y.shape)  # (178,)
print(np.unique(y, return_counts=True))

ohe = OneHotEncoder(sparse=False)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
print(type(x))  # <class 'numpy.ndarray'>
print(type(y))  # <class 'numpy.ndarray'>


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337, stratify=y # stratify는 x와 y를 균등한 비율로 분배
)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

from tensorflow.python.keras.models import load_model
path1 = './_save/keras28_mcp/09_wine/'
model = load_model(path1 + 'k28_0604_1143_0126-0.0718.hdf5')


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
print('f1_score : ', f1)

# loss :  0.10895264893770218
# acc :  0.9444444179534912
# f1_score :  0.945824706694272

# loss :  0.10895264893770218
# acc :  0.9444444179534912
# f1_score :  0.945824706694272

