from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



data_list = [('iris',load_iris(return_X_y=True)),
             ('cancer',load_breast_cancer(return_X_y=True)),
             ('digits',load_digits(return_X_y=True)),
             ('wine',load_wine(return_X_y=True))]

model_list = [LinearSVC(),
              LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]

for name, (x,y) in data_list:
    print(f"\n 데이터셋 : {name}")
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                        random_state=50,
                                                        test_size=0.2)
    for model_set in model_list:
        model = model_set
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        y_predict = model.predict(x_test)
        f1 = f1_score(y_test, y_predict, average='macro')
        print(f'{model.__class__.__name__} F1 score: {score:.4f}')
        
# 데이터셋 : iris
# LinearSVC F1 score: 0.9333
# LogisticRegression F1 score: 0.9333
# DecisionTreeClassifier F1 score: 0.9667
# RandomForestClassifier F1 score: 0.9333

#  데이터셋 : cancer
# LinearSVC F1 score: 0.9737
# LogisticRegression F1 score: 1.0000
# DecisionTreeClassifier F1 score: 0.9298
# RandomForestClassifier F1 score: 0.9561

#  데이터셋 : digits
# LinearSVC F1 score: 0.9500
# LogisticRegression F1 score: 0.9694
# DecisionTreeClassifier F1 score: 0.8389
# RandomForestClassifier F1 score: 0.9750

#  데이터셋 : wine
# LinearSVC F1 score: 1.0000
# LogisticRegression F1 score: 1.0000
# DecisionTreeClassifier F1 score: 0.8889
# RandomForestClassifier F1 score: 0.9444