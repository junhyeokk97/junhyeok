from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
import sklearn as sk
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import random
import numpy as np
import pandas as pd


seed = 123
random.seed(seed)
np.random.seed(seed)

#1.data

datasets  = load_iris()

x=datasets.data
y=datasets.target

print(x.shape,y.shape)

x_tr, x_st, y_tr, y_st = train_test_split(x,y, train_size=0.8, random_state = seed, stratify=y)

#2.모델구성


es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = 'mlogloss',
    data_name='validation_0',
    # save_best=True
)

model=XGBClassifier(
    n_estimators=500,
    max_depth=6,
    gamma = 0,
    min_child_weight=0,
    subsample = 0.4,
    reg_alpha=0,
    reg_lamda=1,
    eval_metrics = 'mlogloss',    #다중분류 : mlogloss, merror, 이진분류 : logloss, error
                                 # 2.1.1버전 이후 핏에서 모델로 위치 이동
    callbacks = [es],                              
    random_state=seed)

model.fit(x_tr, y_tr, 
          eval_set = [(x_st, y_st)],
          verbose=1
          )
print("==========",model.__class__.__name__,"==========")
print('acc : ', model.score(x_st,y_st))
print(model.feature_importances_)   #컬럼별로 어떤게 기여했는지 표기. 피처인포티션
  


print("25퍼 지점 : ",np.percentile(model.feature_importances_,25)) #지점을 찾기 위한 것 


percentile = np.percentile(model.feature_importances_,25)
print(type(percentile))

col_name=[]
#삭제할 역적놈을 찾아내장!
for i, fi in enumerate(model.feature_importances_):
    # print(i,fi)  #잘돌아가는지 체크용
    if fi <= percentile:
        col_name.append(datasets.feature_names[i])
    else: 
        continue

print(col_name)
# ['sepal length (cm)']



print(model.feature_importances_)
# [0.11366749 0.15498699 0.41715884 0.31418666]
thesholds = np.sort(model.feature_importances_)  #오름차순

print(thesholds)
# [0.11366749 0.15498699 0.31418666 0.41715884]

from sklearn.feature_selection import SelectFromModel  #모델을 훈련을 시킬거야 스레스 홀드가 i값 이상인 것을 모두 훈련시킨다.

for i in thesholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)
    
    select_x_tr = selection.transform(x_tr)
    select_x_st = selection.transform(x_st)
    print(select_x_st.shape)
    
    #2.모델구성


    select_model=XGBClassifier(
        n_estimators=500,
        max_depth=6,
        gamma = 0,
        min_child_weight=0,
        subsample = 0.4,
        reg_alpha=0,
        reg_lamda=1,
        eval_metrics = 'mlogloss',    #다중분류 : mlogloss, merror, 이진분류 : logloss, error
                                    # 2.1.1버전 이후 핏에서 모델로 위치 이동
        callbacks = [es],                              
        random_state=seed)

    select_model.fit(select_x_tr, y_tr, 
            eval_set = [(select_x_st, y_st)],
            verbose=1
            )
    
    select_y_prd = select_model.predict(select_x_st)
    score = accuracy_score(y_st, select_y_prd)
  
    print(score)
