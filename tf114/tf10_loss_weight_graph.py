import tensorflow as tf
import matplotlib.pylab as plt

x = [1, 2, 3]
y = [1, 2, 3]
w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y))

w_history = []
loss_history = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict = {w : curr_w})
        
        w_history.append(curr_w)
        loss_history.append(curr_loss)
print("=================== W history =========================")
print(w_history)
print("=================== loss history =========================")
print(loss_history)

import matplotlib.font_manager as fm

path = "C:\\Windows\\Fonts\\NGULIM.TTF"
fontprop = fm.FontProperties(fname=path)
plt.plot(w_history, loss_history)
plt.xlabel('웨이트', fontproperties = fontprop)
plt.ylabel('로스', fontproperties = fontprop)
plt.title('선생님 만세', fontproperties = fontprop)
plt.show()