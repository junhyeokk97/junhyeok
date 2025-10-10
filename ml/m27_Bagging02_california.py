import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier, XGBRegressor
import random
seed = 50
random.seed(seed)
np.random.seed(seed)
from xgboost.callback import EarlyStopping
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,)
                                                    # stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = DecisionTreeRegressor()
# model = BaggingRegressor(DecisionTreeRegressor(),
#                          n_estimators=100,
#                          n_jobs=-1,
#                          random_state=50,
#                          bootstrap=True,    # default
#                          )
# model = RandomForestRegressor(random_state=50)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수: ', results)

# DecisionTreeRegressor
# 최종점수:  0.5658963673701912

# Bagging DecisionTreeRegressor  bootstrap=True     샘플데이터 중복 허용             
# 최종점수:  0.7942234476902508

# Bagging DecisionTreeRegressor   bootstrap=False   샘플데이터 중복 불허
# 최종점수:  0.6032603872566101

# RandomForestRegressor                       
# 최종점수:  0.7948851830424568

