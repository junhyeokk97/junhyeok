import tensorflow as tf
tf.compat.v1.random.set_random_seed(7777)
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission_csv = pd.read_csv(path+'sample_submission.csv')

from sklearn.preprocessing import LabelEncoder
le_geo = LabelEncoder()     # 클래스를 정의화 한다. > 인스턴스화 한다.
le_gen = LabelEncoder()
# train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
le_geo.fit(train_csv['Geography'])
train_csv['Geography'] = le_geo.transform(train_csv['Geography'])

le_gen.fit(train_csv['Gender'])
train_csv['Gender'] = le_gen.transform(train_csv['Gender'])

le_geo.fit(test_csv['Geography'])
test_csv['Geography'] = le_geo.transform(test_csv['Geography'])

le_gen.fit(test_csv['Gender'])
test_csv['Gender'] = le_gen.transform(test_csv['Gender'])

# test_csv['Geography'] = le_geo.fit_transform(test_csv['Geography'])
# test_csv['Gender'] = le_gen.fit_transform(test_csv['Gender'])

print(train_csv['Geography'])
print(train_csv['Geography'].value_counts())
# 0    94215
# 2    36213
# 1    34606
print(train_csv['Gender'])
print(train_csv['Gender'].value_counts())
# 1    93150
# 0    71884


train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)
print(train_csv.columns)    # ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                            #    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                            #    'Exited']

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

# print(x_train.shape)
# print(y_train.shape)
# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.1,
                                                    random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([10,1]), name='weights', dtype=tf.float32)
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
    if step % 20 == 0:
        print(step, 'loss: ', cost_val)


from sklearn.metrics import accuracy_score
y_pred = tf.compat.v1.matmul(tf.cast(x, tf.float32), w_val) + b_val
y_pred = sess.run(y_pred, feed_dict={x: x_test})

y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_pred = (y_pred >= 0.5).astype(int)

sess.close()

acc = accuracy_score(y_test, y_pred)

print(acc)  # 1.0

