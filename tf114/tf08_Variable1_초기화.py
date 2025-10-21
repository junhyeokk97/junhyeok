import tensorflow as tf
tf.random.set_random_seed(777)

변수 = tf.compat.v1.Variable(tf.random_normal([2]), name='weights')
print(변수) # <tf.Variable 'weights:0' shape=(2,) dtype=float32_ref>

# 초기화 첫 번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer()) # 변수 초기화
aaa = sess.run(변수)
print('aaa:',aaa)  # aaa: [ 2.2086694  -0.73225045]
sess.close()

# 초기화 두 번째
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)   # 텐서플로우 데이터 형태인 '변수'를 파이썬에서 쓸 수 있게 바꿔준다.
print('bbb:',bbb)  # bbb: [ 2.2086694  -0.73225045]
sess.close()

# 초기화 세 번째
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()   # ccc: [ 2.2086694  -0.73225045]
print('ccc:',ccc)
sess.close()
