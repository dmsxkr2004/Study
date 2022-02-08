import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
# sess = tf.Session()
sess = tf.compat.v1.Session()

# 실습
# 덧셈 node3
# 뺄셈 node4
# 곱셈 node5
# 나눗셈 node6

# 만들기

# node3 = node1 + node2
print('node1, node2 : ', sess.run([node1, node2]))
node3 = tf.add(node1, node2)
print('node3_더하기 : ', sess.run(node3))

print('node1, node2 : ', sess.run([node1, node2]))
node4 = tf.subtract(node1, node2)
print('node4_빼기 : ', sess.run(node4))

print('node1, node2 : ', sess.run([node1, node2]))
node5 = tf.multiply(node1, node2)
print('node5_곱하기 : ', sess.run(node5))

print('node1, node2 : ', sess.run([node1, node2]))
node6 = tf.divide(node1, node2)
print('node6_나누기 : ', sess.run(node6))

