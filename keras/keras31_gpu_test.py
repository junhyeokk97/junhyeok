import tensorflow as tf
print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU')           # tensorflow 2.7.4 / 2.9.0 = cpu버전
print(gpus)

if gpus:
    print('GPU 있다.')
else:
    print('GPU 없다.')