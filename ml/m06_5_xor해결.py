import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)
#1 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#2. 모델구성
# model = Perceptron()        # linear와 유사
# model = LinearSVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
model.fit(x_data, y_data, epochs=10000)

results = model.evaluate(x_data, y_data)
y_predict = model.predict(x_data)
acc = accuracy_score(y_data, np.round(y_predict))

print('model.score: ', results)
print('acc: ', acc)

# model.score:  [0.7072846293449402, 0.5]
# acc:  0.5