from sklearn.datasets import load_boston
import tensorflow as tf
tf.compat.v1.set_random_seed(66)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MaxAbsScaler,MinMaxScaler,RobustScaler
datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(506,1)

print(x_data.shape, y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size = 0.8, random_state=66, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w6 = tf.compat.v1.Variable(tf.compat.v1.random.normal([13,70]), name = 'weight6')
b6 = tf.compat.v1.Variable(tf.compat.v1.random.normal([70]), name = 'bias6')

input_layer = tf.matmul(x, w6) + b6

w5 = tf.compat.v1.Variable(tf.compat.v1.random.normal([70,55]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.random.normal([55]), name = 'bias5')

hidden_layer4 = tf.matmul(input_layer, w5) + b5

w4 = tf.compat.v1.Variable(tf.compat.v1.random.normal([55,40]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.random.normal([40]), name = 'bias4')

hidden_layer3 = tf.matmul(hidden_layer4, w4) + b4

w3 = tf.compat.v1.Variable(tf.compat.v1.random.normal([40,25]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.random.normal([25]), name = 'bias3')

hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer3, w3) + b3)

w2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([25,10]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([10]), name = 'bias2')

hidden_layer1 = tf.matmul(hidden_layer2, w2) + b2

w1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([10,1]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name = 'bias1')

hypothesis = tf.matmul(hidden_layer1, w1) + b1

#3 - 1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
# train = optimizer.minimize(loss)

#3 - 2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(11501):
    _, loss_val = sess.run([optimizer, loss], feed_dict = {x : x_train, y : y_train})
    if step % 20 == 0:
        print(step,'\t', loss_val)

from sklearn.metrics import r2_score, mean_absolute_error

y_predict = sess.run(hypothesis, feed_dict = {x : x_test})
print("결과값 : ", '\n',y_predict[:5])
r2 = r2_score(y_test, y_predict)
print("r2_score : ", r2)
