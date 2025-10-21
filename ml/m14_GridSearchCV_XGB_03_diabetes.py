import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingClassifier
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
#1. 데이터
x, y = load_diabetes(return_X_y=True)

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

# 최적의 매개변수:  XGBRegressor(base_score=None, booster=None, callbacks=None,
#              colsample_bylevel=None, colsample_bynode=None,  
#              colsample_bytree=None, device=None, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.01, max_bin=None,
#              max_cat_threshold=None, max_cat_to_onehot=None, 
#              max_delta_step=None, max_depth=None, max_leaves=None,
#              min_child_weight=10, missing=nan, monotone_constraints=None,
#              multi_strategy=None, n_estimators=None, n_jobs=None,
#              num_parallel_tree=None, random_state=None, ...) 
# 최적의 파라미터:  {'learning_rate': 0.01, 'min_child_weight': 10}
# best_score:  0.355896577551594
# model.score:  0.3457048084482188
# r2_score:  0.3457048084482188
# time :  13