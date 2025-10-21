import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler

path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()     # 클래스를 정의화 한다. > 인스턴스화 한다.
le_gen = LabelEncoder()
# train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
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

x = train_csv.drop(['Exited'], axis=1)

y = train_csv['Exited']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=50,
                                                    shuffle=True)

std = StandardScaler()
std.fit(x_train)
x_train = std.transform(x_train)
x_test = std.transform(x_test)

n_split = 5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=50)
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=50)


model = MLPClassifier(max_iter=1000)



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
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        #4. 평가, 예측
        y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_pred)
        print('################################')
        print(f'{name}의 cross_val_score   정답률:')
        print(np.round(np.mean(scores),4))
        print(f'{name}의 cross_val_predict 정답률:')
        print(np.round(acc,4))
    except:
        print(name, '은(는) 에러')
        
print('acc: ', scores, '\n평균 acc: ', np.round(np.mean(scores),2))
print('cross_val_predict ACC: ', acc)