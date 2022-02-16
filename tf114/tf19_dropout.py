import tensorflow as tf
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random.normal([2,1]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name = 'bias')

hidden_layer2 = tf.matmul(x , w) + b

layers = tf.nn.dropout(hidden_layer2, keep_prob=0.7)
