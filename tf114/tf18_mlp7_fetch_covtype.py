from sklearn.datasets import fetch_covtype
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

tf.compat.v1.set_random_seed(104)

dataset = fetch_covtype()
x_data = dataset.data
y_data = dataset.target.reshape(-1,1)

ohe = OneHotEncoder()
ohe.fit(y_data)
y_data = ohe.transform(y_data).toarray()

print(x_data.shape)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split (x_data,y_data,train_size = 0.7, random_state=104, shuffle = True)

x = tf.compat.v1.placeholder('float',shape=[None,54])
y = tf.compat.v1.placeholder('float',shape=[None,7])

w = tf.compat.v1.Variable(tf.zeros([54,7]), name = 'weight')
b = tf.compat.v1.Variable(tf.zeros([1,7]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis=1)) # categorical_crossentropy

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(2001):
        _, cos_val, = sess.run([optimizer, loss], feed_dict = {x:x_train,y:y_train})
        if step % 200 ==0:
            print(step, cos_val)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y_acc_test),dtype=tf.float32))
    a = sess.run(accuracy,feed_dict = {x:x_test,y:y_test})
    print("\nacc : ", a)

    y_acc_test = sess.run(tf.argmax(y_test, 1))
    predict = sess.run(tf.argmax(sess.run(hypothesis, feed_dict={x:x_test}), 1))
    acc = accuracy_score(y_acc_test, predict)
    print("accuracy_score : ", acc)
 
'''
acc :  0.6346842
accuracy_score :  0.6346842298512942
'''
 
