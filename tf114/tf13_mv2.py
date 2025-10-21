import tensorflow as tf
import matplotlib.pyplot as plt


# 1. 데이터
# x1_data = [73., 93., 89., 96., 73.]
# x2_data = [80., 88., 91., 98., 66.]
# x3_data = [75., 93., 90., 100., 70.]
# y_data = [152., 185., 180., 196., 142.]

x_data = [[77, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [73, 66, 70]] # (5, 3)
y_data = [[152., 185., 180., 196., 142.]]   # (5, 1)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

# 2. 모델
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1], name='weights'))
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1], name='bias'))

# hypothesis = x * w + b
hypothesis = tf.compat.v1.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0021)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x:x, y: y_data})
    print(step, '\t', cost_val)
    
sess.close()