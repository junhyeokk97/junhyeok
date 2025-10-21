from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
#1data

seed = 42

random.seed(seed)
np.random.seed(seed)

path = './_data/dacon/따릉이/'      # .(점 한개) = 현재 작업폴더 study25

train_csv = pd.read_csv(path + 'train.csv', index_col=0)   # a=b b를 a에 넣겠다. // index_col : 이 컬럼은 인덱스다
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.dropna()       # train_csv에 결측치 데이터를 삭제 처리해라. 결측치 삭제하고 남은 데이터를 반환해서 덮어쓴다

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info())

x = train_csv.drop(['count'], axis=1) 
y = train_csv['count'] 

print(x.shape ,y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed,)
                                                    # stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


es = xgb.callback.EarlyStopping(
    rounds = 50, 
    # metric_name = 'mlogloss',
    data_name = 'validation_0',
    # save_best = True,
    
)

model = XGBRegressor(
    n_estimators = 500,
    random_state = seed,
    gamma = 0,
    min_child_weight = 0,
    reg_alpha = 0,
    reg_lambda = 1,
    # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
    #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
    callbacks = [es]
    
    )


model.fit(x_train, y_train, eval_set =[(x_test,y_test)], verbose=1)




print('r2:', model.score(x_test, y_test))   # acc2: 0.9333333333333333
print(model.feature_importances_)# [0.01230742 0.02487084 0.5794107  0.38341108]


thresholds = np.sort(model.feature_importances_) #오름차순
print(thresholds) #[0.01230742 0.02487084 0.38341108 0.5794107 ]

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit = False)
    # threshold 가 i값 이상인것을 모두 훈련시킨다. 
    # prefit = False : 모델이 아직 학습 되지 않았을때, model.fit 호출해서 훈련한다. (기본)
    # prefit = True : 이미 학습 된 모델을 전달 할 때, model.fit
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape) 

    select_model = XGBRegressor(
        n_estimators = 500,
        random_state = seed,
        gamma = 0,
        min_child_weight = 0,
        reg_alpha = 0,
        reg_lambda = 1,
        # eval_metric = 'mlogloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
        #                         # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
        callbacks = [es]
        
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test,y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, r2: %.4f%%' %(i, select_x_train.shape[1], score*100))
    # print(score)
    # print('acc2:', model.score(select_x_test, y_test))  
        
# Trech=0.027, n=9, r2: 36.0791%
# Trech=0.031, n=8, r2: 36.0749%
# Trech=0.035, n=7, r2: 35.9869%
# Trech=0.043, n=6, r2: 35.8459%
# Trech=0.052, n=5, r2: 35.6012%
# Trech=0.055, n=4, r2: 35.3961%
# Trech=0.140, n=3, r2: 35.1334%
# Trech=0.202, n=2, r2: 32.1168%
# Trech=0.416, n=1, r2: 29.3026%







# print("===============", model.__class__.__name__, "======================")
# print('acc:', model.score(x_test, y_test))      #acc: 0.9333333333333333
# print(model.feature_importances_)


# print('25%지점:', np.percentile(model.feature_importances_, 25))   #0.024616712238639593


# # [0.02430454 0.02472077 0.7376847  0.21328996]
# percentile = np.percentile(model.feature_importances_, 25)
# print(type(percentile))
# # for i, fi in enumerate(model.fe)
# print(xgb.__version__) #내껀 1.7.6 다른 2.1.4


# col_name = []
# #삭제할 컬럼(25% 이하인 놈들) 을 찾아내자!!
# for i, fi in enumerate(model.feature_importances_):
#     # print(i, fi)
#     if fi <= percentile:
#         col_name.append(datasets.feature_names[i])
#     else:
#         continue
    
# print(col_name) #['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# x = pd.DataFrame(x, columns = datasets.feature_names)
# x = x.drop(columns = col_name)

# # print(x)


# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed, stratify=y)

# model.fit(x_train, y_train)

