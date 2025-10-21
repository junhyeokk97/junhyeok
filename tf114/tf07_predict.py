import tensorflow as tf

# 1. 데이터
x_data = [1,2,3,4,5]
y_data = [4,6,8,10,12]
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

x_test_data = [6,7,8]
x_test = tf.placeholder(tf.float32, shape=[None])
#[실습] predict 만들기 / 예상 [14,16,18]

# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(0, dtype=tf.float32)

w = tf.Variable(tf.random_normal([1], dtype=tf.float32))    # random_normal에 들어가는 값은 shape 값을 뜻한다.
b = tf.Variable(tf.random_normal([1], dtype=tf.float32))
print(w)
# <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

# 2. 모델구성
# y = wx + b
hypothesis = x * w + b

# 3-1 컴파일
# model.compile(loss='mse', optimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 3-2 훈련
# sess = tf.compat.v1.Session()
with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    epochs = 1000
    for step in range(epochs):
        # _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:[1,2,3,4,5], y:[4,6,8,10,12]})
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict={x:x_data, y:y_data})
        if step % 20 == 0:
            # print(step, sess.run(loss), sess.run(w), sess.run(b))
            print(step, loss_val, w_val, b_val)
            
# sess.close()
# 4 예측
    # 1
    x_test = tf.placeholder(tf.float32, shape=[None])
    y_predict = x_test * w_val + b_val
    
    results = sess.run(hypothesis, feed_dict={x:x_test_data})
    print(results)
    # [14.035865 16.050875 18.065884]
    
    # 2 파이썬(넘파이) 방식
    y_predict = x_test_data * w_val + b_val
    