import tensorflow as tf
tf.random.set_random_seed(777)

import matplotlib.pyplot as plt

# 1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(0, dtype=tf.float32)

x_test_data = [6,7,8]
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 2. 모델
hypothesis = x * w + b

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse

######################### optimizer #########################
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.021)
# train = optimizer.minimize(loss)

# loss = (hypothesis - y)**2 = (x * w - y)**2
# loss를 w로 미분 = 2 * (x * w - y) * x

lr = 0.02
gradient = tf.reduce_mean((x*w +b -y)*x) # 경사
descent = w - lr * gradient # 하강
train = w.assign(descent) # 기존 w 값을 descent으로 대체.

# 3-2. 훈련
loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    epochs = 2000
    for step in range(epochs):
        _, loss_val, w_val, b_val = sess.run([train,loss,w,b],
                                            feed_dict={x:x_data, y:y_data})
        if step % 100 == 0:
            print(step, loss_val, w_val, b_val)
            
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
    
    print("==========================================")
    y_pred = x_test * w_val + b_val
    print(sess.run(y_pred, feed_dict={x_test:x_test_data}))
    # [13.000007 15.00001  17.000011]
    
plt.plot(loss_val_list)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()