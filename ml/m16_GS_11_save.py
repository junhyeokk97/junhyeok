import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


#1. 데이터 
x,y,= dataset = load_digits(return_X_y=True)
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
print('accuracy_score : ',accuracy_score(y_test,y_pred))

print("걸린 시간:",round(end - start,2),'초')


import joblib
path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_,path + 'm15_best_model11.joblib' )
