from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split#sklearn (사이킷 런)

#1. 데이터
x = np.array(range(1, 17))# 0 ~ 16
y = np.array(range(1, 17))# 0~7, 8~ 13, 14~16
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                random_state=66, shuffle = False , train_size=0.8125)

print(x_train)
print(x_test)
print(y_train)
print(y_test)
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
model.add(Dense(50, input_dim = 1))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs=100, batch_size = 1, 
         # validation_data = (x_val, y_val))
         validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)
