import tensorflow as tf
# tf.compat.v1.random.set_random_seed(7777)
import numpy as np
# 1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.random_normal([2,4]), name='weights')
b = tf.compat.v1.Variable(tf.zeros([4]), name='bias')
# relu 적용 레이어.
# layer1 = tf.nn.relu(tf.matmul(layer, w1) + b1) #여기다 렐루를 적용했다.
# layer1 = tf.nn.selu(tf.matmul(layer, w1) + b1) #여기다 셀루를 적용했다.
layer = tf.nn.elu(tf.matmul(x, w) + b) #여기다 렐루를 적용했다.

rate = tf.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.random.normal([2,1]), name='w2')
b1 = tf.compat.v1.Variable(tf.zeros([1]), name='b2')
layer1 = tf.matmul(layer, w1) + b1


"""드롭 아웃 적용
layer2 = tf.nn.dropout(layer2, rate=0.2)
layer2 = tf.nn.dropout(layer2, keep_prob=0.9)
"""
w2 = tf.compat.v1.Variable(tf.random.normal([1,1]), name='w3')
b2= tf.compat.v1.Variable(tf.zeros([1]), name='b3')

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.matmul(layer1, w2)+b2)

# 3-1. 컴파일
loss = tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 3-2 훈련
epochs = 1010
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b],
                                         feed_dict={x:x_data, y:y_data})

    if step % 20 == 0:
        print(step, 'loss: ', cost_val)

# 4. 평가, 예측
# y_predict = tf.sigmoid(tf.matmul(tf.cast(x_data, tf.float32), w_val) + b_val)

# y_pred = sess.run(tf.cast(y_predict >= 0.5, dtype=tf.float32))
y_pred = sess.run(hypothesis, feed_dict={x: x_data})
from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_pred, y_data)
y_pred_class = (y_pred >= 0.5).astype(int).flatten()
y_true = np.array(y_data).flatten()

acc = accuracy_score(y_true, y_pred_class)


print(acc)  # 0.5