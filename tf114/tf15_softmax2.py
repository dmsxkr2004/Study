import numpy as np
import tensorflow as tf
tf.compat.v1.set_random_seed(66)
from sklearn.model_selection import train_test_split
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1],    # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],    # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],    # 0
          [1, 0, 0]]

x_predict = [[1, 11, 7, 9]]
# print(x_data.shape,y_data.shape)
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 4])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 3])
# x_test = tf.compat.v1.placeholder(tf.float32, shape = [None, 4])
w = tf.compat.v1.Variable(tf.compat.v1.random.normal([4,3]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1,3]), name = 'bias')

#2. 모델구성
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
# hypothesis = tf.sigmoid(hypothesis)
# model.add(Dense(1, activation = 'softmax'))

#3 - 1. 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
# loss = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) # categorical_crossentropy
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.08).minimize(loss)
# train = optimizer.minimize(loss)

#3 - 2. 훈련
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict = {x : x_data, y : y_data})
        if step % 200 == 0:
            print(step, loss_val)
    
    # predict
    results = sess.run(hypothesis, feed_dict = {x:x_data})
    print('\n',results, sess.run(tf.math.argmax(results, 1)))
# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# for i in range(2001):
#     loss_val,hy_val,b_val, _ = sess.run([loss, hypothesis, b, train], feed_dict = {x : x_data, y : y_data})
#     if i % 20 == 0:
#         print(i,'\t','loss : ', loss_val, '\n',hy_val, b_val)

# from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

#4. 평가, 예측

    # y_predict = tf.cast(hypothesis > 0.5, dtype = tf.float32) #tf.cast = 함수안의 조건식이 True 면 1.0 False면 0.0
    # print(sess.run(hypothesis > 0.5, feed_dict = {x : x_data, y : y_data}))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y,y_data),dtype = tf.float32))
    # y_predict_data,acc = sess.run([y_predict,accuracy], feed_dict = {x : x_data, y : y_data})
    # print(sess.run(y_predict))
    # print("=================================")
    # print('예측결과 : \n', y_predict_data)
    print('Accuracy : ', sess.run(accuracy))
    # r2 = r2_score(y_data, y_predict_data)
    # print('r2 : ',r2)

    # mae = mean_absolute_error(y_data, y_predict_data)
    # print('mae : ', mae)

    sess.close()