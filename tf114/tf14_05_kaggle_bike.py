import tensorflow as tf
tf.compat.v1.random.set_random_seed(777)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

path = './_data/kaggle/bike/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)

x = train.drop(['casual', 'registered', 'count'], axis=1)

y = train[['casual', 'registered', 'count']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=190)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 3])


w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 3]), name='weights', dtype=tf.float32)
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias', dtype=tf.float32)

# 2. 모델
hypothesis = tf.compat.v1.matmul(x, w)+b

loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0291)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 100
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_train, y:y_train})
    print(step, '\t', cost_val)
    
# print(type(x_test), x_test.dtype)
# print(type(w_val), w_val.dtype)
    
from sklearn.metrics import r2_score, mean_absolute_error
# # 1. 기존 방법
# y_pred = tf.compat.v1.matmul(tf.cast(x_test, tf.float32), w_val) + b_val
# y_pred = sess.run(y_pred)
# sess.close()

# 2. 재현 방법
y_pred = np.matmul(x_test, w_val) + b_val

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('r2', r2)
print('mae: ', mae)

# r2 -1.2487392514556221
# mae:  10.405640723658543