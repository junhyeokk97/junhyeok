import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

tf.compat.v1.random.set_random_seed(7777)

# 1. 데이터
dataset = load_breast_cancer()

x = dataset.data.astype(np.float32)
y = dataset.target.reshape(-1, 1).astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([30,1]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

# 2. 모델
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# 3-1. 컴파일
loss = -tf.reduce_mean(y * tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))    # binary_crossentropy는 음수 값에서 부터 계산하며 올라가기 때문에 - 를 붙여준다
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-2)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

predict = tf.cast(hypothesis >= 0.5, dtype=tf.float32)
acc_ = tf.reduce_mean(tf.cast(tf.equal(predict, y_train), dtype=tf.float32))

# 3-2. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_train, y:y_train})
    if step % 20 == 0:
        print(step, 'loss: ', cost_val)


from sklearn.metrics import accuracy_score
y_pred = tf.compat.v1.matmul(tf.cast(x_test, tf.float32), w_val) + b_val
y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_pred = (y_pred >= 0.5)

sess.close()

acc = accuracy_score(y_test, y_pred)

print(acc)  # 1.0

