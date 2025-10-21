import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#1 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,1]

#2. 모델구성
model = Perceptron()        # linear와 유사
# model = LinearSVC()
#3. 컴파일, 훈련
model.fit(x_data, y_data)

y_predict = model.predict(x_data)
results = model.score(x_data, y_data)
acc = accuracy_score(y_data, y_predict)

print('model.score: ', results)
print('acc: ', acc)

# model.score:  1.0
# acc:  1.0


# model.score:  1.0
# acc:  1.0