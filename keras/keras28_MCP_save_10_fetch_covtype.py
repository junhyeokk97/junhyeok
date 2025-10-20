
from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]),
# array([211840, 283301,  35754,   2747,   9493,  17367,  20510]

y = pd.get_dummies(y)
print(y)
print(y.shape)  # (581012, 7)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=42)



path = './_save/keras28_mcp/10_fetch_covtype/'
model.save(path + 'keras28_covtype_save.h5')



#4. 평가, 예측
print("=======================================")
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
y_pred = (y_predict > 0.5).astype(int)

acc_score = accuracy_score(y_test, y_predict)

print('loss: ', loss[0])
print('accuracy: ', loss[1])

# loss:  0.28779327869415283
# accuracy:  0.8798320293426514