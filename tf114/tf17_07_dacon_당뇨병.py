import tensorflow as tf
tf.compat.v1.random.set_random_seed(7777)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# 1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']

r = 55
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=r
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([8,1]), name='weights', dtype=tf.float32)
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
acc_ = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

# 3-2. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_train, y:y_train})
    if step % 100 == 0:
        print(step, 'loss: ', cost_val)


from sklearn.metrics import accuracy_score
y_pred = tf.compat.v1.matmul(tf.cast(x, tf.float32), w_val) + b_val
y_pred = sess.run(y_pred, feed_dict={x: x_test})

y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_pred = (y_pred >= 0.5).astype(int)

sess.close()

acc = accuracy_score(y_test, y_pred)

print(acc) # 0.696969696969697

