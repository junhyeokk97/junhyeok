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
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.021)
train = optimizer.minimize(loss)

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
    
print('===========================')
# print(loss_val_list)
# print(w_val_list)

# loss와 eepoch 관계
# plt.plot(loss_val_list)
# plt.show()

# weights와 epoch 관계
# plt.plot(w_val_list)
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('weights')
# plt.show()

# weights와 loss 관계
# plt.plot(w_val_list, loss_val_list)
# plt.xlabel('weights')
# plt.ylabel('loss')
# plt.grid()
# plt.show()

plt.figure(figsize=(15,5))

# (1) loss와 epoch 관계
plt.subplot(1,3,1)
plt.plot(loss_val_list)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()

# (2) weights와 epoch 관계
plt.subplot(1,3,2)
plt.plot(w_val_list)
plt.title("Weights vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Weights")
plt.grid()

# (3) weights와 loss 관계
plt.subplot(1,3,3)
plt.plot(w_val_list, loss_val_list)
plt.title("Weights vs Loss")
plt.xlabel("Weights")
plt.ylabel("Loss")
plt.grid()

plt.tight_layout()
plt.show()