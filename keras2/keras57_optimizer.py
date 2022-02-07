import numpy as np


#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 3, 5, 4, 7, 6, 7, 11, 9, 6])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim =1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
learning_rate = 0.00001
# optimizer = Adam(learning_rate=learning_rate)
# 결과물 :  loss :  2.9373903274536133 lr :  0.0001 결과물 :  [[10.98868]] # learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam', **kwargs # Adam 디폴트
# optimizer = Adadelta(learning_rate=learning_rate)
# loss :  3.2693564891815186 lr :  0.01 결과물 :  [[11.23049]]
# optimizer = Adagrad(learning_rate=learning_rate)
# loss :  4.106063365936279 lr :  0.01 결과물 :  [[11.070601]]
# optimizer = Adamax(learning_rate=learning_rate)
# loss :  3.0176892280578613 lr :  0.0001 결과물 :  [[10.960779]]
# optimizer = RMSprop(learning_rate=learning_rate)
# loss :  3.3632493019104004 lr :  1e-06 결과물 :  [[11.175863]]
# optimizer = SGD(learning_rate=learning_rate)
# loss :  3.3059020042419434 lr :  1e-05 결과물 :  [[11.00326]]
optimizer = Nadam(learning_rate=learning_rate)
# loss :  3.239675521850586 lr :  1e-05 결과물 :  [[10.964725]]

# model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.compile(loss = 'mse', optimizer=optimizer)

model.fit(x,y, epochs=100, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x,y, batch_size=1)
y_predict = model.predict([11])

print('loss : ', loss, 'lr : ',learning_rate, '결과물 : ', y_predict)