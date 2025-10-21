import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier
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
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=55,
                                                    stratify=y)
learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]
max_depth = [3,4,5,6,7]

best_score = 0

for i,lr in enumerate(learning_rate):
    for j,md in enumerate(max_depth):
        model = HistGradientBoostingClassifier(
                learning_rate = lr,
                max_depth = md,)
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        # print(, 'score: ', results)
        if results > best_score:
            best_score = results
            bset_parameters = lr,md
        print(i, ',', j, '번째 score: ', round(results,3), '최고 점수: ', round(best_score,3))

print('최고 점수:  {:.2f}'.format(best_score))
print('최적 매개변수: ', bset_parameters)

# model = HistGradientBoostingClassifier()
# 최고 점수:  0.98
# 최적 매개변수:  (0.1, 6)




