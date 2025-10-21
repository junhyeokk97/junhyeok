import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#1. 데이터
x,y = load_iris(return_X_y=True)

n_split = 5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)
#2. 모델구성
model = MLPClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold) # fit까지 포함.
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))

# acc:  [0.93333333 0.96666667 0.96666667 1.         0.96666667]
# 평균 acc:  0.97

# acc:  [0.94 0.98 0.98] 
# 평균 acc:  0.97