import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

#1. 데이터
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
                                                    random_state=50,
                                                    test_size=0.2)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
test_csv = scaler.fit_transform(test_csv)

#2. 모델구성
from sklearn.svm import LinearSVC
model = LinearSVC(C=0.3)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


model_list= [LinearSVC, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]

for model_set in model_list:
    model = model_set()
    model.fit(x,y)
    score = model.score(x,y)
    y_predict = model.predict(x)
    f1 = f1_score(y, y_predict)
    print(f'{model_set.__name__} F1 score: {score:.4f}')
    
# LinearSVC F1 score: 0.6835
# LogisticRegression F1 score: 0.7856
# DecisionTreeClassifier F1 score: 0.9997
# RandomForestClassifier F1 score: 0.9996