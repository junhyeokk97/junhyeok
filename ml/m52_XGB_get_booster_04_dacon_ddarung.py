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
from sklearn.preprocessing import MinMaxScaler

#1data

seed = 50
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

feature_names = x.columns[]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=seed,)
                                                    # stratify=y)

es = xgb.callback.EarlyStopping(
    rounds = 50, 
    metric_name = 'logloss',
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
    eval_metric = 'logloss', # 다중분류: mlogloss, merror,
                             # 이진분류: logloss, error
                             # 회귀 : rmse, mase, rmsle
                             # 2.1.1 버전 이후로 fit에서 모델로 위치이동
    # callbacks = [es]
    )


model.fit(x_train, y_train, eval_set =[(x_test,y_test)], verbose=1)




print('r2:', model.score(x_test, y_test))   # acc2: 0.9333333333333333
print(model.feature_importances_)# [0.01230742 0.02487084 0.5794107  0.38341108]

# aaa = model.get_booster().get_score(importance_type='weight')     # weight로 설정 시 split 빈도 수 개념
# {'f0': 135.0, 'f1': 95.0, 'f2': 10.0, 'f3': 20.0, 'f4': 43.0, 'f5': 25.0, 'f6': 22.0, 'f7': 66.0, 'f8': 14.0,
#  'f9': 12.0, 'f10': 47.0, 'f11': 40.0, 'f12': 9.0, 'f13': 52.0, 'f14': 25.0, 'f15': 21.0, 'f16': 14.0, 'f17': 10.0,
#  'f18': 11.0, 'f19': 41.0, 'f20': 35.0, 'f21': 79.0, 'f22': 56.0, 'f23': 32.0, 'f24': 52.0, 'f25': 6.0, 'f26': 24.0,
#  'f27': 88.0, 'f28': 70.0, 'f29': 3.0}

aaa = model.get_booster().get_score(importance_type='gain')     # gain로 설정 시 성능 향상량
# {'f0': 0.03657233715057373, 'f1': 0.1393687129020691, 'f2': 0.006830782629549503, 'f3': 0.2773149609565735,
#  'f4': 0.06811584532260895, 'f5': 0.28337687253952026, 'f6': 0.06334161758422852, 'f7': 1.0399045944213867,
#  'f8': 0.015249277465045452, 'f9': 0.050292473286390305, 'f10': 0.10917410999536514, 'f11': 0.5195056796073914,
#  'f12': 0.07042092084884644, 'f13': 0.36393916606903076, 'f14': 0.15158644318580627, 'f15': 0.599240779876709,
#  'f16': 0.42838722467422485, 'f17': 0.17067758738994598, 'f18': 0.13218148052692413, 'f19': 0.12942518293857574,
#  'f20': 9.510327339172363, 'f21': 0.7021641731262207, 'f22': 6.536187648773193, 'f23': 0.6923927068710327, 'f24': 0.16666190326213837,
#  'f25': 0.3431350290775299, 'f26': 0.4401509463787079, 'f27': 1.8994393348693848, 'f28': 0.0855734720826149, 'f29': 0.005506996065378189}
# print(aaa)

total = sum(aaa.values())
print(total)


score_list = [aaa.get(f"f{i}",0) / total for i in range(x.shape[1])]
# 첫 번째 f는 formatted string > f-string 이라고 명칭.
# 두 번째 f는 문자 'f'
# f{i} = f + '0'   >> f0
print(score_list)
print(len(score_list))


# ② 오름차순 정렬
thresholds = np.sort(score_list) #오름차순
'''
####### 컬럼명 매칭 #######
score_df = pd.DataFrame({
    # 'feature' : feature_names ,
    'feature' : [feature_names[int(f[1:])] for f in aaa.keys()],
    'gain' : list(aaa.values())
    }).sort_values(by='gain', ascending=False)   # True 오름차순, False 내림차순
print(score_df)
##################################################
'''
from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    thresholds = np.sort(thresholds)
    selection = SelectFromModel(model, threshold=i, prefit = True)
    # threshold 가 i값 이상인것을 모두 훈련시킨다. 
    # prefit = False : 모델이 아직 학습 되지 않았을때, model.fit 호출해서 훈련한다. (기본)
    # prefit = True : 이미 학습 된 모델을 전달 할 때, model.fit
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape) 

    mask = selection.get_support()
    # print('선택된 피처: ', mask)
    delete_columns = []
    max_acc = 0    
    not_select = [feature_names[j]
           for j, selected in enumerate(mask)
           if not selected]
    
    select_model = XGBRegressor(
        n_estimators = 500,
        random_state = seed,
        gamma = 0,
        min_child_weight = 0,
        reg_alpha = 0,
        reg_lambda = 1,
        eval_metric = 'logloss', # 다중분류: mlogloss, merror, 이진분류: logloss, error
                                # 2.1.1 버전 이후로 fit에서 모델로 위치이동 
        # callbacks = [es]
        
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test,y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_pred)
    print('Trech=%.3f, n=%d, r2: %.4f%%' %(i, select_x_train.shape[1], score*100))

    if max_acc <= score:
        max_acc = score
        delete_columns = not_select
        m = select_x_train.shape[1]
        trech = i
        
    print(delete_columns)
    print('score: ', score)
    print('정규화 gain: ', trech)
    print('삭제할 컬럼: ', m)
    print('')
    # print(score)
    # print('acc2:', model.score(select_x_test, y_test))  
    print('=====================================================')
    
