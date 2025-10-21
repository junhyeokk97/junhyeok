import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

path = 'c:/Study25/_data/kaggle/bike/'

# 데이터
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

# -------------------------
# step 1+2. casual, registered 예측
# -------------------------
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv[['casual', 'registered']] #'registered 예측 추가'
print(x.shape ,y.shape)

print(x.shape, y.shape) #(178, 13) (178,)
print(np.unique(y, return_counts=True))         #(array([0, 1, 2]), array([59, 71, 48]))
# print(pd.value_counts(y))           


# [59, 71, 40]
print(y)


### data 삭제, 라벨이 2인놈을 10개만 남기고 다 지워라!!!!! ###

x = x[:-40]
y = y[:-40]
# print(y)
# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71,  8]))

#traintestsplit은 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, train_size=0.8, shuffle=True, stratify=y)

#######################################SMOTE 적용#############################################
from imblearn.over_sampling import SMOTE, RandomOverSampler
import sklearn as sk

print('sklearn version:', sk.__version__) #sklearn version: 1.6.1

import imblearn
print('imblearn version', imblearn.__version__) #imblearn version 0.13.0 선생님꺼 0.12.4

ros = RandomOverSampler(random_state= seed,
            #RandomOverSampler는 knn을 사용하지 않음.
            # sampling_strategy='auto', #default
            # sampling_strategy= 0.75
              sampling_strategy= {0:5000,1:5000, 2:5000},  #(array([0, 1, 2]), array([50, 57, 33]))
            # n_jobs = -1,  내 버전은 안되고, 선생님 버전은 됨 0.12 이후로 삭제됨/ 이미 포함
            
              )
x_train, y_train = ros.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))


# exit()
#2 model 
model = Sequential()
model.add(Dense(10, input_shape = (13,)))
model.add(Dense(3, activation= 'softmax'))



model.compile(loss = 'sparse_categorical_crossentropy', #원핫 안했잖아!!!!!!!!!!!!!!!
              optimizer= 'adam',
              metrics = ['acc'])


model.fit(x_train, y_train, epochs = 100, validation_split=0.2)


#4 predict, evaluate
result = model.evaluate(x_test, y_test)
print('loss:', result[0])
print('acc:', result[1])


y_pred = model.predict(x_test)
print(y_pred)

# [[6.4038312e-07 9.9998546e-01 1.3949002e-05]
#  [1.1121891e-02 9.8887807e-01 6.3922087e-08]
#  [4.7993505e-19 3.4050580e-02 9.6594942e-01]
#  [4.1012228e-02 9.5898771e-01 1.9110395e-10]
#  [1.0000000e+00 1.3867301e-18 0.0000000e+00]
#  [2.4005217e-22 5.8378184e-01 4.1621813e-01]
# ...
#  [1.0000000e+00 2.1917186e-17 0.0000000e+00]
#  [1.0000000e+00 4.0316573e-13 2.9183994e-34]
#  [1.0000000e+00 6.3728647e-11 9.4042106e-30]
#  [1.0000000e+00 8.3307708e-32 0.0000000e+00]] 이렇게 원핫인코딩 된 상태로 나온다 (35,3)

y_pred = np.argmax(y_pred, axis = 1)
print(y_pred)
print(y_pred.shape) #[1 1 2 1 0 1 1 2 2 0 2 2 0 1 1 0 1 0 0 1 0 1 1 1 0 0 0 0]

acc = accuracy_score(y_pred, y_test)
#f1 다중에서도 사용 가능!!!
f1 = f1_score(y_pred, y_test, average = 'macro')
print('accuracy score:', acc)
print('f1 score:', f1)



###############################결과###################################
#1. 변환하지 않은 원데이터 훈련
# accuracy score: 0.9166666666666666
# f1 score: 0.915079365079365


#2 클래스 2를 40개 삭제한거
# accuracy score: 0.7142857142857143
# f1 score: 0.5496098104793757

#3 SMOTE 
# accuracy score: 0.9285714285714286
# f1 score: 0.8521739130434782


# smote default 
# accuracy score: 0.9285714285714286
# f1 score: 0.8521739130434782