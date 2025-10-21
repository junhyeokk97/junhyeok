from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV 
import time
# 1. 데이터 로드
from sklearn.metrics import mean_squared_error, r2_score
import joblib
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
path = './_save/m15_cv_results/'



model = joblib.load(path + 'm15_best_model5.joblib')


#3. 훈련
# model.fit(x_train,y_train)


# print('최적의 매개변수 : ',model.best_estimator_)
# print('최적의 파라미터 : ',model.best_params_)

#4 평가 예측

# print('best_score : ',model.best_score_)

# Traceback (most recent call last):
#   File "c:\Study25\ml\m17_GS_00_load.py", line 54, in <module>
#     print('best_score : ',model.best_score_)
# AttributeError: 'XGBClassifier' object has no attribute 'best_score_'

# 그리드 서치로 저장했다고 생각하지만 xg부스트의 파라미터만(로) 저장 되어있음 



print('model.score : ',model.score(x_test,y_test))

y_pred= model.predict(x_test)
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))





# 'c:\x5cStudy25\x5cml\x5cm17_GS_05_load.py' ;987ee169-26c2-4b8d-ad21-6580199b6cabmodel.score :  0.3814741300116705
# Mean Squared Error:  20862.635839944232
# R2 Score:  0.3814741300116705