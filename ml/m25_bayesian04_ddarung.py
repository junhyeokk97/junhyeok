import numpy as np
from sklearn.datasets import fetch_california_housing
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

path = './_data/dacon/따릉이/'      # .(점 한개) = 현재 작업폴더 study25

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # a=b b를 a에 넣겠다. // index_col : 이 컬럼은 인덱스다
print(train_csv)                    # [1459 rows x 10 columns] = [1459,10]                  

test_csv = pd.read_csv(path + 'test.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(test_csv)                     # [715 rows x 9 columns] = [715,9]

submission_csv = pd.read_csv(path + 'submission.csv', index_col=0) # 0번째 컬럼을 인덱스로
print(submission_csv)               # [715 rows x 1 columns] = [715,9]  // Nan 데이터가 없음 : 결측치

print(train_csv.shape)              #(1459,10)
print(test_csv.shape)               #(715,9)
print(submission_csv.shape)         #(715,1)

print(train_csv.columns)
                                    # Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                                    #        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                                    #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                                    #       dtype='object')
print(train_csv.info())             # 결측치 확인가능
                                    # 총 1459개인데 컬럼마다 데이터가 빠져있다 데이터를 삭제하면 일부만 없어지더라도 그 행 가로줄이 다 삭제해야한다.
                                    # 최악의 경우 1459에서 하나도 안겹친 데이터가 다 빠져서 많은 데이터가 날라간다.
                                    # 빠진 데이터 위아래값하고 같게 하거나 평균값으로 데이터를 채워도 괜찮다
                                    # 하지만 실제 데이터하고 다를 수 있음 주의 요망 // 결측치 처리에 대한 생각을 많이 해야함.
                                    # 결측치 삭제 or 수정
print(train_csv.describe())

################################ 결측치 처리 1. 삭제 #####################################
print(train_csv.isnull().sum())    # nan=null train_csv의 결측치의 개수를 합해서 출력해라
                                    # hour                        0
                                    # hour_bef_temperature        2
                                    # hour_bef_precipitation      2
                                    # hour_bef_windspeed          9
                                    # hour_bef_humidity           2
                                    # hour_bef_visibility         2
                                    # hour_bef_ozone             76
                                    # hour_bef_pm10              90
                                    # hour_bef_pm2.5            117
                                    # count                       0
                                    # dtype: int64
print(train_csv.isna().sum())       # na=null train_csv의 결측치의 개수를 합해서 출력해라

train_csv = train_csv.dropna()       # train_csv에 결측치 데이터를 삭제 처리해라. 결측치 삭제하고 남은 데이터를 반환해서 덮어쓴다
print(train_csv.isna().sum())        # 결측치 삭제하고 난뒤에 결측치가 다 삭제됐는지 확인
                                     # hour                      0
                                     # hour_bef_temperature      0
                                     # hour_bef_precipitation    0
                                     # hour_bef_windspeed        0
                                     # hour_bef_humidity         0
                                     # hour_bef_visibility       0
                                     # hour_bef_ozone            0
                                     # hour_bef_pm10             0
                                     # hour_bef_pm2.5            0
                                     # count                     0
# print('############################################################')
                                   
print(train_csv.info())              # 1328개의 데이터가 남음 (모든 결측치 삭제)
                                     # <class 'pandas.core.frame.DataFrame'>
                                     # Index: 1328 entries, 3 to 2179
                                     # Data columns (total 10 columns):
                                     #  #   Column                  Non-Null Count  Dtype
                                     # ---  ------                  --------------  -----
                                     #  0   hour                    1328 non-null   int64
                                     #  1   hour_bef_temperature    1328 non-null   float64
                                     #  2   hour_bef_precipitation  1328 non-null   float64
                                     #  3   hour_bef_windspeed      1328 non-null   float64
                                     #  4   hour_bef_humidity       1328 non-null   float64
                                     #  5   hour_bef_visibility     1328 non-null   float64
                                     #  6   hour_bef_ozone          1328 non-null   float64
                                     #  7   hour_bef_pm10           1328 non-null   float64
                                     #  8   hour_bef_pm2.5          1328 non-null   float64
                                     #  9   count                   1328 non-null   float64
                                     # dtypes: float64(9), int64(1)
print(train_csv)                     # [1328 rows x 10 columns]

################################ 테스트 데이터에 결측치 #####################################
print(test_csv.info())
test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

#################### 받은 데이터를 전처리 (결측치 정리) ######################

#################### x,y 데이터를 분리한다 ##########################

x = train_csv.drop(['count'], axis=1) # 축1번 count 컬럼을 삭제하고 남은 데이터를 x에 넣는다
                                      # count 라는 axis=1열 삭제, 참고로 행은 axis=0 (2차원)
                                      # train_csv에서 9개 컬럼만 복사해서 x에 넣어준거지 train_csv의 상태는 그대로다
                                      # x= count 제외한 데이터 y= count 데이터

print(x)                              # [1459 rows x 9 columns]                                

y = train_csv['count']               # count컬럼만 빼서 y에 넣겠다.

print(y.shape) # (1459,)
x = np.array(x)
y = np.array(y)
x = x.reshape(x.shape[0], x.shape[1])  # (1459, 9) >> (1459, 9)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,)
                                                    # stratify=y)

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

# {'target': 0.763489463570841, 'params': {'n_estimators': 297.60945888926034, 'learning_rate': 0.5, 'max_depth': 7.287873455800198, 'min_child_weight': 7.568666292857121, 'gamma': 0.0, 'subsample': 2.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 1.0, 'reg_lambda': 66.89305759858338, 'reg_alpha': 3.5188844183787245}}
# r2_score:  0.763489463570841
# 걸린 시간:  160.48