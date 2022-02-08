# y = wx + b

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.set_random_seed(66)

#1. 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(1, dtype = tf.float32)
b = tf.Variable(1, dtype = tf.float32)

#2. 모델구성
hypothesis = x_train * W + b    # y = wx + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01, use_locking=False, name="GradientDescent")
train = optimizer.minimize(loss)    #optimizer = 'adam'
# model.compile(loss = 'mse', optimizer = 'sgd')
#3-2 훈련
# sess = tf.compat.v1.Session()

with tf.compat.v1.Session() as sess: # with를 쓰는이유는 sess를 사용하고 close를 해줘야 sess가 종료되는데 with를 쓰면 close를 쓰지않아도 됨
    
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0: # 20의 나머지가 0과 같을때 출력한다.
            print(step, sess.run(loss), sess.run(W),sess.run(b))
    
#4. 평가, 예측


