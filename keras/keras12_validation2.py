from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1, 17))# 0 ~ 16
y = np.array(range(1, 17))# 0~7, 8~ 13, 14~16
x_train = x[1:8]
y_train = y[1:8]
x_test = x[8:14]
y_test = y[8:14]
x_val = x[14:17]
y_val = y[14:17]

'''
x_train = np.array(range(10))
y_train = np.array(range(10))
x_test = np.array([11, 12, 13])
y_test = np.array([11, 12, 13])
x_val = np.array([14, 15, 16])
y_val = np.array([14, 15, 16])
'''

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs=100, batch_size = 1, 
          validation_data = (x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)
