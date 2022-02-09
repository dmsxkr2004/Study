import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)
                            # input_dim = 1
변수 = tf.Variable(tf.compat.v1.random_normal([1]))
print(변수)

#1.
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(변수)
print("aaa : ",aaa) # aaa :  [0.06524777]
sess.close()

#2.
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = 변수.eval(session=sess)
print("bbb : ", bbb)
sess.close()

#3.
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = 변수.eval()
print("ccc : ", ccc)
sess.close()