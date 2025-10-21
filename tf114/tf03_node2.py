import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

# 실습
# 덧셈, 뺄셈(subtract), 곱셈(multiply), 나눗셈(divide)

node3 = tf.add(node1, node2)
sess = tf.compat.v1.Session()
print(sess.run(node3)) # 5.0

node3 = tf.subtract(node1, node2)
sess = tf.compat.v1.Session()
print(sess.run(node3)) # -1.0

node3 = tf.multiply(node1, node2)
sess = tf.compat.v1.Session()
print(sess.run(node3)) # 6.0

node3 = tf.divide(node1, node2)
sess = tf.compat.v1.Session()
print(sess.run(node3)) # 0.6666667