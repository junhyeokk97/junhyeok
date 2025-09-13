from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV 
import time
import joblib
# 1. 데이터 로드

import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

path = './_data/dacon/따릉이/'   
train_csv = pd.read_csv(path + 'train.csv', index_col=0)  
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
train_csv = train_csv.fillna(train_csv.mean())



############## test ################

test_csv = test_csv.fillna(test_csv.mean())



x = train_csv.drop(['count'], axis=1)   
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


# 5. 모델 정의
#2. 모델
xgb = XGBRegressor()
model = GridSearchCV(xgb, parameters,cv = kfold, # 54 * 5 = 270번 
                     verbose = 1,
                     refit=True, # 1번
                     n_jobs = -1, # 271번 

                     )

start = time.time()

#3. 훈련
model.fit(x_train,y_train)

end =time.time()


print('최적의 매개변수 : ',model.best_estimator_)
print('최적의 파라미터 : ',model.best_params_)

#4 평가 에측

print('best_score : ',model.best_score_)
print('model.score : ',model.score(x_test,y_test))

y_pred= model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print('r2 score : ',r2)
print("걸린 시간:",round(end - start,2),'초')



path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_,path + 'm15_best_model4.joblib' )
