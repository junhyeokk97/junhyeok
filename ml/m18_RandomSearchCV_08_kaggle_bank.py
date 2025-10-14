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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import joblib
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
#1. 데이터
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

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_test = std.transform(x_test)

n_split = 5

kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12],
     'learning_rate':[0.1 , 0.01, 0.001]},  # 18
    {'max_depth':[6,8,10,12], 'learning_rate':[0.1 , 0.01, 0.001]}, # 12
    {'min_child_weight':[2,3,4,5,10], 'learning_rate':[0.1 , 0.01, 0.001]}  # 15
]

#2. 모델
xgb = XGBClassifier()

model = RandomizedSearchCV(xgb, parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=-1,
                     random_state=50,
                     n_iter=11)

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
print('accuracy_score: ', accuracy_score(y_test, y_pred))
print('time : ', round(end-str))

# 최적의 매개변수:  XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None, 
#               colsample_bytree=None, device=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=0.1, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,              max_delta_step=None, max_depth=None, max_leaves=None,
#               min_child_weight=3, missing=nan, monotone_constraints=None,
#               multi_strategy=None, n_estimators=None, n_jobs=None,
#               num_parallel_tree=None, random_state=None, ...)최적의 파라미터:  {'learning_rate': 0.1, 'min_child_weight': 
# 3}
# best_score:  0.8662167755134981
# model.score:  0.8630593510467477
# accuracy_score:  0.8630593510467477
# time :  196

path = './_save/m15_cv_results/'
pd.DataFrame(model.cv_results_).sort_values('rank_test_score',    # rank_test_score 기준으로 오름차순 정렬.
                                                  ascending=True).to_csv(path + 'm18_rs_08_kaggle_bank_cv_results.csv')

path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_, path + 'm18_08_best_model.joblib')  # 가중치 저장.


# 최적의 파라미터:  {'max_depth': 6, 'learning_rate': 0.1}
# best_score:  0.8658077706743041
# model.score:  0.8633017238767534
# accuracy_score:  0.8633017238767534
# time :  35