import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=50,
                                                    test_size=0.2)

std = StandardScaler()
x = std.fit_transform(x)

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
    
# LinearSVC F1 score: 0.7684
# LogisticRegression F1 score: 0.7699
# DecisionTreeClassifier F1 score: 1.0000
# RandomForestClassifier F1 score: 1.0000