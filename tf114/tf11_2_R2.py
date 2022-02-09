from numpy import gradient
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.set_random_seed(66)
x_train_data = [1, 2, 3]
y_train_data = [1, 2, 3]

x_test_data = [4, 5, 6]
y_test_data = [4, 5, 6]

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

x_test = tf.compat.v1.placeholder(tf.float32)
y_test = tf.compat.v1.placeholder(tf.float32)
w = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]), name = 'weight')

hypothesis = x * w

loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y))#reduce_mean

lr = 0.1
grad = tf.reduce_mean((w * x - y) * x)
descent = w - lr * grad
update = w.assign(descent)      # w = w - lr * gradient

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# w_history = []
# loss_history = []
print("Epochs\tLoss\t\tWeight")
for i in range(21):
    # sess.run(update, feed_dict={x : x_train, y : y_train})
    # print(i, '\t', sess.run(loss, feed_dict={x : x_train, y : y_train})), sess.run(w)
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict = {x : x_train_data, y : y_train_data})
    print(i, '\t', loss_v,'\t', w_v)

from sklearn.metrics import r2_score, mean_absolute_error

y_predict =  x_test * w_v
y_predict_data = sess.run(y_predict, feed_dict={x_test : x_test_data, y_test : y_test_data})
print('y_predict : ', y_predict_data)

r2 = r2_score(y_test_data, y_predict_data)
print('r2 : ', r2)

mae = mean_absolute_error(y_test_data, y_predict_data)
print('mae : ', mae)
sess.close()
