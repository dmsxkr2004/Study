import tensorflow as tf

tf.compat.v1.set_random_seed(66)

x_data = [[73,51,65],
          [92,98,11],
          [89,31,33],
          [99,33,100],
          [17,66,79]]

y_data = [[152],[185],[180],[205],[142]]

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 3])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.compat.v1.random.normal([3,1]), name = 'weight')
b = tf.compat.v1.Variable(tf.random.normal([1]), name = 'bias')

hypothesis = tf.matmul(x, w) + b

#3 - 1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(loss)

#3 - 2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for i in range(2001):
    _, loss_val, w_val, b_val = sess.run([train, loss, w, b], feed_dict = {x : x_data, y : y_data})
    if i % 20 == 0:
        print(i,'\t', loss_val, w_val, b_val)

from sklearn.metrics import r2_score, mean_absolute_error

y_predict = tf.matmul(x , w_val) + b_val
y_predict_data = sess.run(y_predict, feed_dict = {x : x_data, y : y_data})
print('y_predict : ', y_predict_data)

r2 = r2_score(y_data, y_predict_data)
print('r2 : ',r2)

mae = mean_absolute_error(y_data, y_predict_data)
print('mae : ', mae)

sess.close()
