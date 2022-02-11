import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
tf.compat.v1.set_random_seed(66)

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape,y_data)
y_data = y_data.reshape(569,1)
# x_train,x_test,y_train,y_test = train_test_split(x_data,y_data, random_state=66, train_size=0.8, shuffle=True)
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.zeros([30,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias')

#2. 모델구성
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
# hypothesis = tf.sigmoid(hypothesis)
# model.add(Dense(1, activation = 'sigmoid'))

#3 - 1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 8e-9).minimize(loss)
# train = optimizer.minimize(loss)

#3 - 2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(2001):
    _, loss_val, b_val = sess.run([optimizer, hypothesis, b], feed_dict = {x : x_data, y : y_data})
    if i % 200 == 0:
        print(i,'\t','loss : ', loss_val, '\n', b_val)

from sklearn.metrics import r2_score, mean_absolute_error

#4. 평가, 예측
y_predict = tf.cast(hypothesis > 0.5, dtype = tf.float32) #tf.cast = 함수안의 조건식이 True 면 1.0 False면 0.0
# print(sess.run(hypothesis > 0.5, feed_dict = {x : x_data, y : y_data}))
accuracy = tf.reduce_min(tf.cast(tf.equal(y,y_predict),dtype = tf.float32))
y_predict_data,acc = sess.run([y_predict,accuracy], feed_dict = {x : x_data, y : y_data})

print("=================================")
print('예측결과 : \n', y_predict_data[:5])
print('Accuracy : ', acc)

# r2 = r2_score(y_data, y_predict_data)
# print('r2 : ',r2)

# mae = mean_absolute_error(y_data, y_predict_data)
# print('mae : ', mae)

sess.close()
'''
=================================
예측값 : 
 [[0.04392562]
 [0.17360151]
 [0.36277783]
 [0.7560311 ]
 [0.92249364]
 [0.9746191 ]]
예측결과 :  
[[0.]
[0.]
[0.]
[1.]
[1.]
[1.]]
Accuracy :  1.0
'''