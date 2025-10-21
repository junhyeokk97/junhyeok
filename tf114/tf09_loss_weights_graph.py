import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.compat.v1.placeholder(tf.float32)

# 2. 모델
hypothesis = x * w

# 3. 컴파일, 훈련
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})

        w_history.append(curr_w)
        loss_history.append(curr_loss)

print('=========== w history ===========')
print(w_history)

print('=========== loss history ===========')
print(loss_history)

plt.plot(w_history, loss_history)
plt.xlabel('weights')
plt.ylabel('loss')
plt.grid()
plt.show()