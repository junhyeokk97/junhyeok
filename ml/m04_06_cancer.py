from sklearn.metrics import f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
dataset = load_breast_cancer()
x,y = load_breast_cancer(return_X_y=True)

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

# LinearSVC F1 score: 0.9877
# LogisticRegression F1 score: 0.9877
# DecisionTreeClassifier F1 score: 1.0000
# RandomForestClassifier F1 score: 1.0000




