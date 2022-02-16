import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D

tf.compat.v1.set_random_seed(66)

#1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from keras.utils import to_categorical # 1부터 시작한다.
# one hot은 0부터
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/165
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/165

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. 모델구성

w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 128]) # Conv2D에서 kernel size 역할을 한다.
L1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding = 'SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides = [1, 2, 2, 1], padding= 'SAME')

# model.add(conv2D(filters = 64, kernel_size = (2,2), strides=(1,1), padding = 'SAME', input_shape = (28, 28, 1))) # 위의 두줄과 같다.

print(w1) # <tf.Variable 'w1:0' shape=(2, 2, 1, 64) dtype=float32_ref>
print(L1) # Tensor("Conv2D:0", shape=(?, 28, 28, 64), dtype=float32)
print(L1_maxpool) # Tensor("MaxPool:0", shape=(?, 14, 14, 64), dtype=float32)

# Layer 2
w2 = tf.compat.v1.get_variable('w2', shape = [3, 3, 128, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides = [1,1,1,1], padding = 'SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
print(L2_maxpool)       # (?, 7, 7, 64)

# Layer 3
w3 = tf.compat.v1.get_variable('w3', shape = [3, 3, 64, 32])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides = [1,1,1,1], padding = 'SAME')
L3 = tf.nn.selu(L3)
L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
print(L3_maxpool)       # (?, 4, 4, 64)

# Layer 4
w4 = tf.compat.v1.get_variable('w4', shape = [3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides = [1,1,1,1], padding = 'SAME')
L4 = tf.nn.selu(L4)
L4_maxpool = tf.nn.max_pool2d(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'SAME')
print(L4_maxpool)       # (?, 4, 4, 64)

# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*32]) # 플래튼 :  Tensor("Reshape:0", shape=(?, 128), dtype=float32)
print("플래튼 : ", L_flat)

# layer 5 DNN

w5 = tf.compat.v1.get_variable('w5', shape = [128, 64], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.compat.v1.random.normal([64]), name = 'b5')
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
L5 = tf.nn.dropout(L5, keep_prob=0.7)

w6 = tf.compat.v1.get_variable('w6', shape = [64,32], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.compat.v1.random.normal([32]), name = 'b6')
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.relu(L6)
L6 = tf.nn.dropout(L6, keep_prob=0.7)

w7 = tf.compat.v1.get_variable('w7', shape = [32,10], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.compat.v1.random.normal([10]), name = 'b7')
L7 = tf.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)
print(hypothesis)

#3 - 1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
######################################################################
training_epochs = 10
batch_size = 100
total_batch = int(len(x_train)/batch_size)
print(total_batch)
######################################################################

#3 - 2. 훈련
for epoch in range(training_epochs):
    avg_loss = 0
    
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start : end], y_train[start : end]
        
        feed_dict = {x:batch_x, y: batch_y}
        
        batch_loss, _ = sess.run([loss, optimizer], feed_dict = feed_dict)
        
        avg_loss += batch_loss / total_batch
    print('Epochs : ', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))
print("훈련 끝")

#4. 평가, 예측
from sklearn.metrics import accuracy_score
y_predict = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(y_predict, tf.float32))
print('ACC : ', sess.run(accuracy, feed_dict = {x : x_test, y : y_test}))


# y_predict = sess.run(hypothesis, feed_dict = {x : x_test})
# print("결과값 : ", '\n',y_predict[:5])
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)