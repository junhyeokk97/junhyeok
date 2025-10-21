from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV
import time
# 1. 데이터 로드

import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

path = './_data/kaggle/bike/'    #절대경로
                                                                                 

train_csv = pd.read_csv(path + 'train.csv', index_col=0) # columns이 index로 바뀜
test_csv = pd.read_csv(path + 'test.csv', index_col=0)




x = train_csv.drop(['casual', 'registered', 'count'], axis=1) 
y = train_csv['count']


# 3. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=55)

# 4. 스케일링
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



n_split = 5 
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

parameters =[
    {'n_estimators' : [100,500],'max_depth':[6,10,12],'learning_rate':[0.1,0.01,0.001]}, # 18
    {'max_depth':[6,8,10,12],'learning_rate':[0.1,0.01,0.001]}, #12
    {'min_child_weight':[2,3,5,10],'learning_rate':[0.1,0.01,0.001]} # 12
]


#2. 모델
xgb = XGBRegressor()
model = RandomizedSearchCV(xgb, parameters,cv = kfold, # 54 * 5 = 270번 
                     verbose = 1,
                     refit=True, # 1번
                     n_jobs = -1, # 271번 
                     n_iter=11, # 디폴트 10
                     random_state=33

                     )

start = time.time()

#3. 훈련
model.fit(x_train,y_train)

end =time.time()

print('최적의 매개변수 : ',model.best_estimator_)
print('최적의 파라미터 : ',model.best_params_)

# exit() # Fitting 5 folds for each of 11 candidates, totalling 55 fits


#4 평가 에측

print('best_score : ',model.best_score_)
print('model.score : ',model.score(x_test,y_test))

print('best_score : ',model.best_score_)
print('model.score : ',model.score(x_test,y_test))

y_pred= model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2 score : ',r2)
print("걸린 시간:",round(end - start,2),'초')

path = './_save/m15_cv_results/'

print(pd.DataFrame(model.cv_results_).sort_values(
    'rank_test_score',
    ascending = True).to_csv(path + 'm18_00_rs_cv_05.csv',index = False)) 


import joblib

joblib.dump(model.best_estimator_,path + 'm18_05_best_model.joblib' )
# model 이 아닌 model.best_estimator_ 로 저장한 이유는 model로 저장하면 gridsearch의 자잘한 파라미터까지 전부 포함 귀찮아짐 

# best_score :  0.3378826401703842
# model.score :  0.37441151023313435
# best_score :  0.3378826401703842
# model.score :  0.37441151023313435
# r2 score :  0.37441151023313435