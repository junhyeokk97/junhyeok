import tensorflow as tf
tf.compat.v1.random.set_random_seed(7777)

# 1. 데이터
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([2,1]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

# 2. 모델
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 3-1. 컴파일
loss = -tf.reduce_mean(y * tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))    # binary_crossentropy는 음수 값에서 부터 계산하며 올라가기 때문에 - 를 붙여준다
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

predict = tf.cast(hypothesis >= 0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

# 3-2. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss: ', cost_val)
acc = sess.run(acc, feed_dict={x:x_data,y:y_data})

print('최종 w: ', w_val, 'b: ', b_val)
print('acc: ', acc) # acc:  1.0
