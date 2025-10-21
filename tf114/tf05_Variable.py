import tensorflow as tf
print(tf.__version__)
sess = tf.compat.v1.Session()

a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([1], dtype=tf.float32)

# print(sess.run(a +b))

init = tf.compat.v1.global_variables_initializer()
sess.run(init)

print(sess.run(a+ b))   # [3.]