import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle
import joblib
#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=34,
                                                    test_size=0.2,
                                                    stratify=y)
       
parameters ={'n_estimators': 1000,
              'learning_rate': 0.3,
              'max_depth': 3,
              'gamma': 1,
              'min_child_weight': 1,
              'subsample': 1,
              'colsample_btree': 1,
              'colsample_bylevel': 1,
              'colsamople_bynode': 1,
              'reg_alpha': 0,
              'reg_lambda': 1,
              'random_state': 33,
              'verbose': 2,}
                                                    
#2. 모델구성
path = './_save/m01_job/'
model = pickle.load(open(path + 'm02_pickle_save.pickle'))

results = model.score(x_test, y_test)
print('점수:' ,results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc:', acc)


# joblib.dump(model, path + 'm01_joblib_save.joblib')
# pickle.dump(model, path + 'm02_pickle_save.pickle')