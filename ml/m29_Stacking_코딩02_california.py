import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

x,y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=50)

xgb = XGBRegressor(verbose=0)
rf = RandomForestRegressor(verbose=0)
cat = CatBoostRegressor(verbose=0)
lg = LGBMRegressor(verbose=0)

models = [xgb, cat, lg]

train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)     # stacking 사용 시, 나온 값으로 다시 학습 시키기 위해서 
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)     # y_train_pred에서 나온 예측 값들로 train_list를 만들어 준다.
    test_list.append(y_test_pred)
    
    score = r2_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} r2: {1: .4f}'.format(class_name, score))     # 첫 번째 변수가 0 자리에, 두 번째 변수가 1 자리에 들어간다.

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
model2 = CatBoostRegressor(verbose=0)
model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred2)
print('stacking: ', score2)