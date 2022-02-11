from sklearn.datasets import load_boston
import tensorflow as tf
tf.compat.v1.set_random_seed(66)
import numpy as np

datasets = load_boston()
x_data = datasets.data
y_data = datasets.target
y_data = y_data.reshape(506,1)
print(x_data, y_data)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 13])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random.normal([13,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델구성
hypothesis = tf.matmul(x, w) + b

#3 - 1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-6)
train = optimizer.minimize(loss)

#3 - 2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(200001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict = {x : x_data, y : y_data})
    if i % 2000 == 0:
        print(i,'\t', loss_val, w_val, b_val)

from sklearn.metrics import r2_score, mean_absolute_error

#4. 평가, 예측
y_predict = tf.matmul(x , w_val) + b_val
y_predict_data = sess.run(y_predict, feed_dict = {x : x_data, y : y_data})
print('y_predict : ', y_predict_data[0:5])

r2 = r2_score(y_data, y_predict_data)
print('r2 : ',r2)

mae = mean_absolute_error(y_data, y_predict_data)
print('mae : ', mae)
