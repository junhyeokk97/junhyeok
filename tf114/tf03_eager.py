import tensorflow as tf
print('tf version: ', tf.__version__)
print('즉시실행모드: ', tf.executing_eagerly())
# tf version:  1.14.0
# 즉시실행모드:  False

# tf version:  2.7.4
# 즉시실행모드:  True

tf.compat.v1.disable_eager_execution()
print('즉시실행모드: ', tf.executing_eagerly())
# 즉시실행모드:  False

# tf.compat.v1.enable_eager_execution()
# print('즉시실행모드: ', tf.executing_eagerly())
# # 즉시실행모드:  True

hello = tf.constant("Hello world!!")
sess = tf.compat.v1.Session()
print(sess.run(hello))     # 즉시실행모드에선 error         /       tensor1 에서는 즉시실행모드 on/off 중요
                           # 즉시실행모드를 끄고 실행.

###########################################
# 즉시 실행모드 > tensor1의 그래프 형태의 구성 없이 자연스러운 파이썬 문법으로 실행
# tf.compat.v1.disable_eager_execution() # 즉시 실행 모드 끄기 // tensorflow 1.0 문법(기본)
# tf.compat.v1.enable_eager_execution()  # 즉시 실행 모드 켜기 // tensorflow 2.0 사용 가능

# sess.run()
# 가상환경              즉시 실행 모드              사용가능
#  1.14.0               disable(기본)              b'Hello world!'
#  1.14.0               ensable                    error
#  2.7.4                disable(기본)              b'Hello world!'
#  2.7.4                ensable                    error
    
"""
tensor1은 그래프 연산 모드
tensor2는 즉시 실행 모드

tf.compat.v1.enable_eager_execution()      # 즉시 실행 모드 켬
> tensor2의 기본

tf.compat.v1.disable_eager_execution()      # 즉시 실행 모드 끔
> 그래프 연산 모드로 돌아감
> tensor1 코드 사용 가능

tf.executing_eagerly()
> True : 즉시실행모드, tensor2 코드만 사용 가능
> False : 그래프 연산 모드, tensor1 코드 가능
"""