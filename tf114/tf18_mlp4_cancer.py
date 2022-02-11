import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
tf.compat.v1.set_random_seed(66)

#1. 데이터
datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target
print(x_data.shape,y_data.shape)
y_data = y_data.reshape(569,1)
# x_train,x_test,y_train,y_test = train_test_split(x_data,y_data, random_state=66, train_size=0.8, shuffle=True)
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.zeros([30,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name = 'bias')

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size = 0.8, random_state=66, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


w6 = tf.compat.v1.Variable(tf.compat.v1.random.normal([30,70]), name = 'weight6')
b6 = tf.compat.v1.Variable(tf.compat.v1.random.normal([70]), name = 'bias6')

input_layer = tf.matmul(x, w6) + b6

w5 = tf.compat.v1.Variable(tf.compat.v1.random.normal([70,55]), name = 'weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.random.normal([55]), name = 'bias5')

hidden_layer4 = tf.nn.relu(tf.matmul(input_layer, w5) + b5)

w4 = tf.compat.v1.Variable(tf.compat.v1.random.normal([55,40]), name = 'weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.random.normal([40]), name = 'bias4')

hidden_layer3 = tf.nn.relu(tf.matmul(hidden_layer4, w4) + b4)

w3 = tf.compat.v1.Variable(tf.compat.v1.random.normal([40,25]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.random.normal([25]), name = 'bias3')

hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer3, w3) + b3)

w2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([25,10]), name = 'weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.random.normal([10]), name = 'bias2')

hidden_layer1 = tf.matmul(hidden_layer2, w2) + b2

w1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([10,1]), name = 'weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name = 'bias1')

hypothesis = tf.nn.sigmoid(tf.matmul(hidden_layer1, w1) + b1)

#3 - 1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.000000001).minimize(loss)
# train = optimizer.minimize(loss)

#3 - 2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(2001):
    _, loss_val = sess.run([optimizer, loss], feed_dict = {x : x_train, y : y_train})
    if i % 200 == 0:
        print(i,'\t','loss : ', loss_val)

from sklearn.metrics import r2_score, mean_absolute_error,accuracy_score

#4. 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x : x_test}) #tf.cast = 함수안의 조건식이 True 면 1.0 False면 0.0
# print(sess.run(hypothesis > 0.5, feed_dict = {x : x_data, y : y_data}))
# accuracy = tf.reduce_min(tf.cast(tf.equal(y,y_predict),dtype = tf.float32))
# y_predict_data,acc = sess.run([y_predict,accuracy], feed_dict = {x : x_data, y : y_data})
acc = accuracy_score(y_test, y_predict)
print("=================================")
print('예측결과 : \n', y_predict[:5])

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