import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()     # 클래스를 정의화 한다. > 인스턴스화 한다.
le_gen = LabelEncoder()

le_geo.fit(train_csv['Geography'])
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])

le_gen.fit(train_csv['Gender'])
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])

le_geo.fit(test_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(test_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])




train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)
print(train_csv.columns)
x = train_csv.drop(['Exited'], axis=1)
print(x.shape)  # (165034, 10)
y = train_csv['Exited']
print(y.shape)  # (165034,)

x = x.to_numpy().reshape(x.shape[0], x.shape[1])  # (165034, 10)
y = y.to_numpy().reshape(y.shape[0], 1)  # (165034, 1)

x_train, x_test, y_train, y_test= train_test_split(x,y,
                                                   random_state=50,
                                                   test_size=0.2)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#2. 모델 구성
# model = RandomForestRegressor()
allAlgorithms = all_estimators(type_filter='classifier')
print('allAlgorithms: ', allAlgorithms)
print('모델: ', len(allAlgorithms))     # 55
print(type(allAlgorithms))

max_score = 0
max_name = 'name'
for (name, algorithm) in allAlgorithms:
    ######## 예외 처리 #######      try , except
    try:
        model = algorithm()
        
        #3. 훈련
        model.fit(x_train, y_train)
        #4. 평가, 예측
        results = model.score(x_test, y_test)
        print(name, 'score: ', results)
        if results > max_score:
            max_score = results
            max_name = name
    except:
        print(name, '은(는) 에러')
        
print("max model: ", max_name, max_score)

# max model:  HistGradientBoostingClassifier 0.8647256642530372