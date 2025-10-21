# xgb   california, diabetes,
# lgbm  cancer, digits

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV

# 1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, shuffle=True, random_state=333, stratify=y)


parameters = [
    {'randomforestclassifier__min_samples_split__n_estimators': [100, 200], 'randomforestclassifier__min_samples_split__max_depth': [5,6,10], 'randomforestclassifier__min_samples_split__min_samples_leaf': [3,10]},
    {'randomforestclassifier__min_samples_split__max_depth': [6,8,10,12], 'randomforestclassifier__min_samples_split__min_samples_leaf': [3,5,7,10]},
    {'randomforestclassifier__min_samples_split__min_samples_leaf': [3,5,7,9], 'randomforestclassifier__min_samples_split__min_samples_split': [2,3,5,10]},
    {'randomforestclassifier__min_samples_split': [2,3,5,6]}]
# scl = MinMaxScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# # 2. 모델
# model = RandomForestClassifier()
# model = make_pipeline(StandardScaler(), RandomForestClassifier())
# model = make_pipeline(MinMaxScaler(), SVC())
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())        # make_pipeline의 파라미터는 풀네임으로 지정해야함.
# pipe = Pipeline([('std', StandardScaler()), ('rf', RandomForestClassifier())])
model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('score: ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)


# score:  0.8666666666666667
# acc:  0.8666666666666667