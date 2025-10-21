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
import warnings
warnings.filterwarnings('ignore')
import joblib
#1. 데이터
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

n_split = 5

# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)
kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12],
     'learning_rate':[0.1 , 0.01, 0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1 , 0.01, 0.001]}, # 12
    {'min_child_weight':[2,3,4,5,10], 'learning_rate':[0.1 , 0.01, 0.001]}  # 15
]

#2. 모델
# xgb = XGBClassifier()
xgb = XGBRegressor()
model = GridSearchCV(xgb, parameters, cv=kfold,
                     verbose=1,
                     refit=True,    # test split 되지 않은 상태로 전체 테스트 1번 진행
                     n_jobs=-1)     # 총 train 271번

#3. 훈련
str = time.time()
model.fit(x_train, y_train)
end = time.time()

print('최적의 매개변수: ', model.best_estimator_)
print('최적의 파라미터: ', model.best_params_)

#4. 평가, 예측
print('best_score: ', model.best_score_)    # train에서의 최고 성능.
print('model.score: ', model.score(x_test,y_test))

y_pred = model.predict(x_test)
print('r2_score: ', r2_score(y_test, y_pred))
print('time : ', round(end-str))

path = './_save/m15_cv_results'
joblib.dump(model.best_estimator_, path+ 'm15_best_model_03.joblib')

# 최적의 매개변수:  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.1, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=6, max_leaves=None,
#              min_child_weight=None, missing=nan, monotone_constraints=None,
#              multi_strategy=None, n_estimators=100, n_jobs=None,
#              num_parallel_tree=None, random_state=None, ...)
# 최적의 파라미터:  {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
# best_score:  0.780609408677946
# model.score:  0.7489860421142809
# r2_score:  0.7489860421142809
# time :  27