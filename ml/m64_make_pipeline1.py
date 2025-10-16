import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# 1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, shuffle=True, random_state=333, stratify=y)

# scl = MinMaxScaler()
# x_train = scl.fit_transform(x_train)
# x_test = scl.transform(x_test)

# # 2. 모델
# model = RandomForestClassifier()

model = make_pipeline(StandardScaler(), RandomForestClassifier())
model = make_pipeline(MinMaxScaler(), SVC())

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
results = model.score(x_test, y_test)
print('score: ', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)