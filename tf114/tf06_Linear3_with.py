import tensorflow as tf

# 1. 데이터
x = [1,2,3,4,5]
y = [4,6,8,10,12]

w = tf.Variable(111, dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

# 2. 모델구성
# y = wx + b
hypothesis = x * w + b

# 3-1 컴파일
# model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0308)
train = optimizer.minimize(loss)

# 3-2 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 100
    for step in range(epochs):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b))
    # sess.close()
