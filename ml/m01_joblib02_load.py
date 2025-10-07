import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=34,
                                                    test_size=0.2,
                                                    stratify=y)

                                                    
#2. 모델구성, #3. 훈련 - 불러오기

path = './_save/m01_job/'
model = joblib.load(path + 'm01_joblib_save.joblib')

#4. 평가, 예측
results = model.score(x_test, y_test)
print('점수:' ,results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc:', acc)


# joblib.dump(model, path + 'm01_joblib_save.joblib')
# 점수: 0.9210526315789473
# acc: 0.9210526315789473