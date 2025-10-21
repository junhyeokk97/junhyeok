import tensorflow as tf
print(tf.__version__)
# 1.14
# node1 = tf.constant(3.0)
# node2 = tf.constant(4.0)
# node3 = node1 + node2

node1 = tf.compat.v1.placeholder(tf.float32)
node2 = tf.compat.v1.placeholder(tf.float32)
node3 = node1 + node2

sess = tf.compat.v1.Session()
# print(sess.run(node3))
print(sess.run(node3, feed_dict={node1:3, node2:4}))    # 7.0
# print(sess.run(node3, feed_dict={node1:10, node2:17}))  # 27.0

node3_triple = node3 * 3
print(node3_triple)
print(sess.run(node3_triple), feed_dict={node1:3, node2:4}) # 21.0

