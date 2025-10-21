import numpy as np

from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
# x = datasets.data
# y = datasets['target']
x, y = load_iris(return_X_y=True)
print(x.shape, y.shape)
# (150, 4), (150, )



# #2. 모델구성
# from sklearn.svm import LinearSVC
# model = LinearSVC(C=0.3)
# 0.96

# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# 0.9733333333333334

# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# 1.0

# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# 1.0


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(x, y , epochs=10)
model.fit(x,y)

results = model.score(x, y)
print(results)