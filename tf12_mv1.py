import tensorflow as tf
tf.compat.v1.set_random_seed(66)
         # 첫번 둘번 세번 네번  다섯번
x1_data = [73., 93., 89., 96., 73.]        # 국어
x2_data = [80., 88., 91., 98., 66.]        # 영어
x3_data = [75., 93., 90., 100., 70.]       # 수학
y_data = [152., 185., 180., 196., 142.]    # 환산점수

# x는 (5,3), y는 (5,1)또는 (5,)
# y = x1 * w1 + x2 * w2 + x3 * w3

x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)


w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight1')
w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight2')
w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='weight3')
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name='bias')


# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())
# print(sess.run([w1, w2, w3]))

#2. 모델
hypothesis = x1*w1 + x2*w2 + x3*w3 + b

#3 - 1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(loss)

#3 - 2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(2001):
    _, loss_val, [w1_v, w2_v, w3_v], b_val = sess.run([train, loss, [w1, w2, w3], b], feed_dict = {x1 : x1_data, x2 : x2_data, x3 : x3_data, y : y_data})
    if i % 20 == 0:
        print(i,'\t', loss_val,[w1_v, w2_v, w3_v], b_val)



from sklearn.metrics import r2_score, mean_absolute_error

y_predict = x1*w1 + x2*w2 + x3*w3 + b_val
y_predict_data = sess.run(y_predict, feed_dict = {x1 : x1_data, x2 : x2_data, x3 : x3_data, y : y_data})
print('y_predict : ', y_predict_data)

r2 = r2_score(y_data, y_predict_data)
print('r2 : ', r2)

mae = mean_absolute_error(y_data, y_predict_data)
print('mae : ', mae)
sess.close()