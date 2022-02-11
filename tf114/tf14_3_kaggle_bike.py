import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
# def RMSE(y_test, y_pred):
#     return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터
path = '../_data/kaggle/bike/'
train = pd.read_csv(path+'train.csv')
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')

x_data = train.drop(['datetime', 'casual','registered','count'], axis=1)
test_file = test_file.drop(['datetime',], axis=1)
y_data = train['count']
# y = np.log1p(y)
print(x_data.shape,y_data.shape)
y_data = y_data.values.reshape(10886, 1)
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random.normal([8,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

#2. 모델구성
hypothesis = tf.matmul(x, w) + b

#3 - 1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-7)
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

'''
r2 :  0.21807365491006914
mae :  119.22677262099475
'''