import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)
print(tf.executing_eagerly()) # True = 즉시 실행모드

tf.compat.v1.disable_eager_execution() # 끄는것

print(tf.executing_eagerly())

hello = tf.constant("Hello World!")

sess = tf.compat.v1.Session()
print(sess.run(hello))

