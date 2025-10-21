import tensorflow as tf
import matplotlib.pyplot as plt


# 1. 데이터
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

x1 = tf.placeholder(tf.float32, shape=[None])
x2 = tf.placeholder(tf.float32, shape=[None])
x3 = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

# 2. 모델
w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
b = tf.compat.v1.Variable([0], dtype=tf.float32, name='bias')

hypothesis = x1*w1 + x2*w2 + x3*w3 +b

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0021)
train = optimizer.minimize(loss)

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     epochs = 1000
#     for step in range(epochs):
#         _, loss_val, w1_val, w2_val, w3_val, b_val = sess.run([train, loss, w1, w2, w3, b],
#                                                               feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})
#         if step % 100 == 0:
#             print(step, loss_val, w1_val, b_val)
#             print(step, loss_val, w2_val, b_val)
#             print(step, loss_val, w3_val, b_val)
            
#     pred = sess.run(hypothesis, feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
#     print("pred:", pred)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
for step in range(epochs):
    cost_val, _ = sess.run([loss, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, y: y_data})
    print(step, '\t', cost_val)
    
sess.close()