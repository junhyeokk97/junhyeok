import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder     # 다중분류 : OneHotEncoder 무조건
tf.random.set_random_seed(777)

# 1. 데이터
from sklearn.datasets import load_iris
datasets = load_iris()
x = datasets.data
y = datasets.target

y = y.reshape(-1,1)
OHE=OneHotEncoder(sparse=False)
OHE.fit(y)
y = OHE.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42,test_size=0.1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, y_train.shape)

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])

w = tf.compat.v1.Variable(tf.random_normal([4,3]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias', dtype=tf.float32)

# 2. 모델 구성
hypothesis = tf.nn.softmax(tf.matmul(x, w)+b)

# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1))  # softmax / categorical_crossentropy

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

from sklearn.metrics import accuracy_score
predict = tf.argmax(hypothesis, 1)
y_pred = tf.equal(predict, tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(y_pred, tf.float32))

# 3-2. 훈련
epochs = 4001
for step in range(epochs):
    cost_val, _, w_val, b_val, acc_val = sess.run([loss, train, w, b, acc], feed_dict={x:x_test, y:y_test})
    if step % 20 == 0:
        print(step, 'loss:', cost_val, 'acc:',acc_val)


sess.close()


# 4000 loss: 0.040058527 acc: 1.0