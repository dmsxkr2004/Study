from numpy import gradient
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(77)
x_train = [1, 2, 3]
y_train = [1, 2, 3]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([1]), name = 'weight')


hypothesis = x * w # [1, 2, 3] * 0
                    #행렬평균값                제곱
loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y)) # (0 - [1, 2, 3])

lr = 0.1
grad = tf.reduce_mean((w * x - y) * x) # (0 * [1,2,3]-[1,2,3]) * [1,2,3]) = -4.666667 = grad
descent = w - lr * grad # 0 - 0.2 * -4.6666667 = 0.9333333 = descent
update = w.assign(descent)      # w = w - lr * gradient 

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())#변수 전체 초기화
# print(update)
w_history = []
loss_history = []
print("Epochs\tLoss\t\tWeight")

for i in range(21):
    # sess.run(update, feed_dict={x : x_train, y : y_train})
    # print(i, '\t', sess.run(loss, feed_dict={x : x_train, y : y_train})), sess.run(w)
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict = {x : x_train, y : y_train})
    print(i,'\t', loss_v,'\t', w_v)
    w_history.append(w_v)
    loss_history.append(loss_v)

import matplotlib.font_manager as fm

path = "C:\\Windows\\Fonts\\NGULIM.TTF"
fontprop = fm.FontProperties(fname=path)
plt.plot(w_history, loss_history)
plt.xlabel('웨이트', fontproperties = fontprop)
plt.ylabel('로스', fontproperties = fontprop)
plt.title('선생님 만세', fontproperties = fontprop)
plt.show()

