import numpy as np
from sklearn.datasets import load_digits
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
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=seed,
                                                    stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()
# model = BaggingClassifier(DecisionTreeClassifier(),
#                          n_estimators=100,
#                          n_jobs=-1,
#                          random_state=50,
#                          bootstrap=False,    # True = default
#                          )
model = RandomForestClassifier(random_state=50)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수: ', results)

# DecisionTreeClassifier
# 최종점수:  0.85

# Bagging DecisionTreeClassifier      bootstrap=True
# 최종점수:  0.9611111111111111

# Bagging DecisionTreeClassifier      bootstrap=False
# 최종점수:  0.8944444444444445

# RandomForestClassifier
# 최종점수:  0.9777777777777777