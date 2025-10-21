import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import r2_score
import pandas as pd 


#1. 데이터 
x,y,= dataset = load_wine(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(
    x,y,train_size=0.8,random_state=55,stratify=y
)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


n_split = 5
kfold = StratifiedKFold(n_splits = n_split, shuffle = True, random_state = 55)


parameters =[
    {'n_estimators' : [100,500],'max_depth':[6,10,12],'learning_rate':[0.1,0.01,0.001]}, # 12
    {'max_depth':[6,8,10,12],'learning_rate':[0.1,0.01,0.001]},
    {'min_child_weight':[2,3,5,10],'learning_rate':[0.1,0.01,0.001]}
]




#2. 모델
xgb = XGBClassifier()
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

y_pred= model.predict(x_test)
print('accuracy_score : ',accuracy_score(y_test,y_pred))

y_pred_best = model.best_estimator_.predict(x_test)
print('best_acc_score :', accuracy_score(y_test, y_pred_best))

# 위나 아래나 별 차이 없음 쓰고 싶은거 쓰세요;;

print("걸린 시간:",round(end - start,2),'초')

path = './_save/m15_cv_results/'

print(pd.DataFrame(model.cv_results_).sort_values(
    'rank_test_score',
    ascending = True).to_csv(path + 'm18_00_rs_cv_09.csv',index = False)) 


import joblib

joblib.dump(model.best_estimator_,path + 'm18_09_best_model.joblib' )
# model 이 아닌 model.best_estimator_ 로 저장한 이유는 model로 저장하면 gridsearch의 자잘한 파라미터까지 전부 포함 귀찮아짐 


# best_score :  0.9645320197044336
# model.score :  0.9722222222222222
# accuracy_score :  0.9722222222222222
# best_acc_score : 0.9722222222222222
# 걸린 시간: 4.13 초