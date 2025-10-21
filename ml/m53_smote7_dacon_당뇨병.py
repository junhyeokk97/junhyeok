import numpy as np
import pandas as pd
import random

from sklearn.datasets import load_breast_cancer
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
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

print(x_train.shape, y_train.shape) #(586, 8) (586,)
print(np.unique(y, return_counts=True))        # (array([0, 1], array([424, 228],
# print(pd.value_counts(y))           


# [59, 71, 40]
print(y)


### data 삭제, 라벨이 2인놈을 10개만 남기고 다 지워라!!!!! ###

# x = x[:-40]
# y = y[:-40]
# print(y)
#traintestsplit은 
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, train_size=0.8, shuffle=True, stratify=y)

#######################################SMOTE 적용#############################################
from imblearn.over_sampling import SMOTE
import sklearn as sk
import imblearn

print('sklearn version:', sk.__version__) #sklearn version: 1.6.1
print('imblearn version', imblearn.__version__) #imblearn version 0.13.0 선생님꺼 0.12.4

smote = SMOTE(random_state= seed,
              k_neighbors=2,        #default
              sampling_strategy='auto', #default
            # sampling_strategy= 0.75
              # sampling_strategy= {0:5000, 1:5000},  #(array([0, 1, 2]), array([50, 57, 33]))
              n_jobs = -1, # 내 버전은 안되고, 선생님 버전은 됨 0.12 이후로 삭제됨/ 이미 포함
            
              )
x_train, y_train = smote.fit_resample(x_train, y_train)

print(np.unique(y_train, return_counts=True))


# exit()
#2 model 
model = Sequential()
model.add(Dense(10, input_shape = (8,)))
model.add(Dense(1, activation= 'sigmoid'))



model.compile(loss = 'binary_crossentropy', #원핫 안했잖아!!!!!!!!!!!!!!!
              optimizer= 'adam',
              metrics = ['acc'])


model.fit(x_train, y_train, epochs = 100, validation_split=0.2)


#4 predict, evaluate
result = model.evaluate(x_test, y_test)
print('loss:', result[0])
print('acc:', result[1])


y_pred = model.predict(x_test)
print(y_pred)
y_pred = np.round(y_pred)




# y_pred = np.argmax(y_pred, axis = 1)
print(y_pred)
print(y_pred.shape) #[1 1 2 1 0 1 1 2 2 0 2 2 0 1 1 0 1 0 0 1 0 1 1 1 0 0 0 0]

acc = accuracy_score(y_pred, y_test)
#f1 다중에서도 사용 가능!!!
f1 = f1_score(y_pred, y_test)
print('accuracy score:', acc)
print('f1 score:', f1)



###############################결과###################################
#3 SMOTE 
# accuracy score: 0.8015267175572519
# f1 score: 0.6666666666666666


# smote default 
# accuracy score: 0.6793893129770993
# f1 score: 0.5882352941176471