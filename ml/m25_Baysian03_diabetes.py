import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')
import random
from tensorflow.keras.callbacks import EarlyStopping


seed =333
random.seed(seed)
np.random.seed(seed)


#1 데이터
from sklearn.datasets import load_diabetes

x, y = load_diabetes(return_X_y=True)

# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=55)

n_split = 5



scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델 구성 
bayesian_params = {
               'n_estimators' :(100,500),
               'learning_rate' : (0.001,0.1),
               'max_depth' : (3,10),
            #    'num_leaves' : (24,40),
            #    'min_child_samples' : (10,200),
               'min_child_weight' : (1,50),
               'gamma' :(0,5),
               'subsample' : (0.5,1), # 0~1 사이 값만 줘야함
               'colsample_bytree' : (0.5,1),
               'colsample_bylevel' : (0.5,1),
            #    'max_bin' : (9,500),
               'reg_lambda' : (0,100), # 디폴트 1 // L2 정규화 //릿지
               'reg_alpha' : (0,10)    # 디폴트 0 // L1 정규화 //라쏘          
               }


def xgb_hamsu(n_estimators,learning_rate, max_depth,min_child_weight,gamma,subsample,colsample_bytree,colsample_bylevel,reg_lambda,reg_alpha):
    params = {
               'n_estimators' : int(round(n_estimators)),
               'learning_rate' : learning_rate,
               'max_depth' : int(round(max_depth)),
               'min_child_weight' : int(round(min_child_weight)),
               'gamma' : int(round(gamma)),
               'subsample' : max(min(subsample,1),0),
               'colsample_bytree' : colsample_bytree,
               'colsample_bylevel' : colsample_bylevel,
               'reg_lambda' : max(reg_lambda,0),
               'reg_alpha' : reg_alpha              
            }
    
    
    
    
    model = XGBRegressor(**params,early_stopping_rounds=10,  eval_metric='rmse',n_jobs = -1)
    model.fit(x_train,y_train, eval_set = [(x_test,y_test)],           
              verbose = 0,)
    
    y_pred = model.predict(x_test)
    result = r2_score(y_test,y_pred)
    return result
             
                


optimizer = BayesianOptimization(
    f = xgb_hamsu,
    pbounds = bayesian_params,
    random_state=333,
    verbose=2
)
n_iter = 300
start = time.time()
optimizer.maximize(init_points = 20, n_iter=n_iter)
end = time.time()

print(optimizer.max)
print(n_iter,'번 걸린 시간:',round( end - start),'초')

# {'target': 0.849554897605619, 
# 'params': {'n_estimators': 457.5294698918733, 
# 'learning_rate': 0.1, 'max_depth': 10.0, 
# 'min_child_weight': 17.102784071951767, 
# 'gamma': 0.0, 'subsmaple': 1.0, 'colsample_bytree': 0.5, 
# 'colsample_bylevel': 1.0, 'reg_lambda': 21.467398499890976, 'reg_alpha': 3.5741362812352397}}
# 300 번 걸린 시간: 349 초
