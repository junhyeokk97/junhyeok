import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
seed = 123
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1 data
dataset = load_digits()
x = dataset.data
y = dataset.target

print(x.shape, y.shape) #(569, 30) (569,)
print(np.unique(y, return_counts=True))        # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180],
# print(pd.value_counts(y))           

# [59, 71, 40]
print(y)

### data 삭제, 라벨이 2인놈을 10개만 남기고 다 지워라!!!!! ###


#traintestsplit은 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, train_size=0.9, shuffle=True, stratify=y)

#######################################SMOTE 적용#############################################
from imblearn.over_sampling import SMOTE
import sklearn as sk
import imblearn

smote = SMOTE(random_state= seed,
              k_neighbors=3,        #default
              #sampling_strategy='auto', #default
            # sampling_strategy= 0.75
              sampling_strategy= {0:500, 1:500, 2:500, 3:500, 4:500, 5:500, 6:500, 7:500, 8:500, 9:500},  #(array([0, 1, 2]), array([50, 57, 33]))
              n_jobs = -1, # 내 버전은 안되고, 선생님 버전은 됨 0.12 이후로 삭제됨/ 이미 포함
            
              )
x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))


# exit()
#2 model 
model = Sequential()
model.add(Dense(30, input_shape = (64,)))
model.add(Dense(15))
model.add(Dense(10, activation= 'softmax'))



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


#3 SMOTE 
# accuracy score: 0.9611111111111111
# f1 score: 0.9610337316529268


# smote default 
# accuracy score: 0.9666666666666667
# f1 score: 0.9664950664950667