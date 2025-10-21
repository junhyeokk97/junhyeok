import tensorflow as tf
print(tf.__version__)

## 텐서플로우 설치 오류시 - 250829
# pip install protobuf==3.20
# pip install tensorflow==1.16

print("hello world")

hello = tf.constant("hello world")
print(hello)    # Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))  # b'hello world', b = binary, 그래프 연산을 실행시킴.