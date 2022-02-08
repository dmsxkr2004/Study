import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype = tf.int32)

init = tf.compat.v1.global_variables_initializer()
sess.run(init)

print("잘 나오니 ", sess.run(x))

