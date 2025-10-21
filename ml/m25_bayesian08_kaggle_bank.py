import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import joblib
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()     # 클래스를 정의화 한다. > 인스턴스화 한다.
le_gen = LabelEncoder()
# train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
le_geo.fit(train_csv['Geography'])
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])

le_gen.fit(train_csv['Gender'])
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])

le_geo.fit(test_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(test_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)

y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,
                                                    stratify=y)

bayesian_params = {
    'n_estimators' : (100, 500),
    'learning_rate' : (0.0001, 0.5),
    'max_depth' : (3, 10),
    # 'num_leaves' : (24, 40),
    # 'min_child_samples' : (10, 200),
    'min_child_weight' : (1, 10),
    'gamma' : (0, 5),
    'subsample' : (0.5, 2),     # 0~1 사이에.
    'colsample_bytree' : (0.5, 1),
    'colsample_bylevel' : (0.5, 1),
    # 'max_bin' : (9, 500),
    'reg_lambda' : (0, 100),
    'reg_alpha' : (0, 10)
} 

def xgb_hamsu(n_estimators, learning_rate, max_depth, min_child_weight,gamma 
              ,subsample, colsample_bytree, colsample_bylevel, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : int(n_estimators),
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'min_child_weight' : int(round(min_child_weight)),
        'gamma' : gamma,
        'subsample' : max(min(subsample,1),0),       # 0~1 사이로 형성.
        'colsample_bytree' : colsample_bytree,
        'colsample_bylevel' : colsample_bylevel,
        'reg_lambda' : max(reg_lambda, 0),      # dafault : 1 // L2 정규화 //릿지
        'reg_alpha' : reg_alpha                 # default : 0 // L1 정규화 // 라쏘
    }
    model = XGBRegressor(**params, n_jobs=-1)
    
    
    
    model.fit(x_train, y_train,
              eval_set = [(x_test, y_test)],
              verbose=0,
              eval_metric='rmse',
              early_stopping_rounds=15)
    y_pred = model.predict(x_test)
    results = r2_score(y_test, y_pred)
    return results

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = xgb_hamsu,     # 블랙박스 함수.
    pbounds=bayesian_params,
    random_state=50,
)

n_iter = 100
str = time.time()
optimizer.maximize(init_points=5,    # init_point >> 최적화 시작 지점 또는 초기 시도 횟수
                   n_iter=n_iter)
end = time.time()
print(optimizer.max)
print('r2_score: ', optimizer.max['target'])
print('걸린 시간: ', round(end-str,2))

# {'target': 0.4046397805213928, 'params': {'n_estimators': 471.3415795180606, 'learning_rate': 0.06900260154043722, 'max_depth': 9.148329694921802, 'min_child_weight': 6.220919304970101, 'gamma': 0.27636365735082435, 'subsample': 1.5919554455452336, 'colsample_bytree': 0.8710761615445917, 'colsample_bylevel': 0.7804027565966098, 'reg_lambda': 94.50904559778085, 'reg_alpha': 5.953294153196795}}
# r2_score:  0.4046397805213928
# 걸린 시간:  490.23