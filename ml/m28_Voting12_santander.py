import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
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
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import VotingRegressor, VotingClassifier
import random
seed = 50
random.seed(seed)
np.random.seed(seed)
from xgboost.callback import EarlyStopping
path = './_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

x = train_csv.drop(columns=['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,
                                                    stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

xgb = XGBClassifier()
lg = LGBMClassifier()
cat = CatBoostClassifier()

#2. 모델
model = VotingClassifier(
    estimators=[('XGB', xgb),('lg', lg),('cat', cat)],
    # voting = 'hard',    # dafault
    voting = 'soft'
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수: ', results)

# 최종점수:  0.9161     hard
# 최종점수:  0.916825   soft