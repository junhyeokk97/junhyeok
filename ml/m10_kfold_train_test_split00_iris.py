import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
                        
#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,
                                                    shuffle=True,
                                                    stratify=y)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)

#2. 모델구성
model = MLPClassifier()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc: ', scores, '\n 평균 acc: ', np.round(np.mean(scores),2))

# acc:  [0.95833333 0.95833333 0.95833333 1.         1.        ] 
#  평균 acc:  0.98

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)
print(y_pred)
# [0 2 1 1 2 2 2 1 0 0 1 0 1 1 1 0 1 0 0 0 0 1 2 1 2 2 0 2 2 2]
# [0 2 1 1 2 2 2 1 0 0 1 0 1 1 1 0 1 0 0 0 0 1 2 1 2 2 0 2 2 2]

acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC: ', acc)
# cross_val_predict ACC:  1.0