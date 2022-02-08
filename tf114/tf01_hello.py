import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# print('Hello world!!')
hello = tf.constant("Hello world!")

print(hello)
# Tensor("Const:0", shape=(), dtype=string)

#tf.constant
#tf.variable
#tf.placeholder

sess = tf.compat.v1.Session()
print(sess.run(hello))

