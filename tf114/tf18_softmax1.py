import tensorflow as tf
import numpy as np
tf.random.set_random_seed(777)

# 1. 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]
y_data = [[0,0,1],
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

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
    cost_val, _, w_val, b_val, acc_val = sess.run([loss, train, w, b, acc], feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss:', cost_val, 'acc:',acc_val)


sess.close()


# 4000 loss: 0.29044056 acc: 0.8755