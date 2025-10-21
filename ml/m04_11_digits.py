from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.datasets import load_digits

dataset = load_digits()
x,y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=50,
                                                    test_size=0.2)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
scaler.fit(x_test)
x_test = scaler.fit_transform(x_test)

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
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    y_predict = model.predict(x_test)
    y_predict = np.round(y_predict)
    f1 = f1_score(y_test, y_predict, average='macro')
    print(f'{model_set.__name__} F1 score: {score:.4f}')
    
    