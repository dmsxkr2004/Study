# 실습
# lr 수정해서 epochs 100번 이하로 줄여라
# step = 100이하, w = 1.9999, b = 0.9999

# y = wx + b

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(77)

#1. 데이터
x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# W = tf.Variable(2, dtype = tf.float32)
# b = tf.Variable(0, dtype = tf.float32)
W = tf.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32)
b = tf.Variable(tf.compat.v1.random_normal([1]), dtype = tf.float32)

#2. 모델구성
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(W))
hypothesis = x_train * W + b    # y = wx + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1679)
train = optimizer.minimize(loss)    #optimizer = 'adam'
# model.compile(loss = 'mse', optimizer = 'sgd')

#3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101):
    # sess.run(train)
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict = {x_train : [1, 2, 3], y_train : [3, 5, 7]})
    if step % 1 == 0: # 20의 나머지가 0과 같을때 출력한다.
        # print(step, sess.run(loss), sess.run(W),sess.run(b))
        print(step, loss_val, W_val, b_val)


#4. 평가, 예측
# sess.run(tf.compat.v1.global_variables_initializer())
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_predict =  x_test * W_val + b_val
print(sess.run(y_predict, feed_dict = {x_test : [6,7,8]}))
sess.close()