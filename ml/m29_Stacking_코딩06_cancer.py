import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50,
                                                    stratify=y)

xgb = XGBClassifier(verbose=0)
rf = RandomForestClassifier(verbose=0)
cat = CatBoostClassifier(verbose=0)
lg = LGBMClassifier(verbose=0)

models = [xgb, rf, cat, lg]

train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)     # stacking 사용 시, 나온 값으로 다시 학습 시키기 위해서 
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)     # y_train_pred에서 나온 예측 값들로 train_list를 만들어 준다.
    test_list.append(y_test_pred)
    
    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} acc: {1: .4f}'.format(class_name, score))     # 첫 번째 변수가 0 자리에, 두 번째 변수가 1 자리에 들어간다.

x_train_new = np.array(train_list).T
# print(x_train_new)
# [[1.53734457 1.55549    1.62230979 1.73895468]
#  [3.02624512 2.9829002  3.11070224 3.20784045]
#  [1.00209737 1.06836    1.07577524 0.98158987]
#  ...
#  [1.69604003 1.8309401  1.99070386 1.87717334]
#  [1.69461381 1.81621    1.72278086 1.61583267]
#  [1.43430364 1.66825    1.37041036 1.36039316]]
# print(x_train_new.shape)
# (18576, 4)

x_test_new = np.array(test_list).T

#2-2 모델
model2 = CatBoostClassifier(verbose=0)
model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)
score2 = accuracy_score(y_test, y_pred2)
print('stacking: ', score2)


# XGBClassifier acc:  0.9825
# RandomForestClassifier acc:  0.9649
# CatBoostClassifier acc:  0.9825
# LGBMClassifier acc:  1.0000
# stacking:  0.9824561403508771