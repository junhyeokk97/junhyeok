import tensorflow as tf
tf.compat.v1.random.set_random_seed(777)
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder 
# 1. 데이터
from sklearn.datasets import fetch_covtype
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

y = y.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42,test_size=0.1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 54])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random_normal([54,20]), name='weights')
b = tf.compat.v1.Variable(tf.zeros([20]), name='bias')
layer = tf.matmul(x, w) + b

w1 = tf.compat.v1.Variable(tf.random.normal([20,10]), name='w1')
b1 = tf.compat.v1.Variable(tf.zeros([10]), name='b1')
layer1 = tf.matmul(layer, w1) + b1

w2 = tf.compat.v1.Variable(tf.random.normal([10,4]), name='w2')
b2 = tf.compat.v1.Variable(tf.zeros([4]), name='b2')
layer2 = tf.matmul(layer1, w2) + b2
"""드롭 아웃 적용
layer2 = tf.nn.dropout(layer2, rate=0.2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.9)
"""
w3 = tf.compat.v1.Variable(tf.random.normal([4,1]), name='w3')
b3= tf.compat.v1.Variable(tf.zeros([1]), name='b3')

# 2. 모델
hypothesis = tf.nn.softmax(tf.matmul(layer2, w3)+b3)

# 3-1. 컴파일
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 3-2 훈련
epochs = 100
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x:x_train, y:y_train})

    if step % 20 == 0:
        print(step, 'loss: ', cost_val)

# 4. 평가, 예측
# y_predict = tf.sigmoid(tf.matmul(tf.cast(x_data, tf.float32), w_val) + b_val)

# y_pred = sess.run(tf.cast(y_predict >= 0.5, dtype=tf.float32))
y_pred = sess.run(hypothesis, feed_dict={x: x_test})
from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_pred, y_data)
y_predict = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
acc = accuracy_score(y_true, y_predict)


print(acc)  # 1.0