# https://yujuwon.tistory.com/entry/NumPy 넘파이 함수 정의 정리사이트
import numpy as np # 넘파이 함수를 np 로 정의하겠다
from tensorflow.keras.models import Sequential # Sequential 함수를 불러오겟다
from tensorflow.keras.layers import Dense # Dense 함수를 불러오겠다

#1. 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7])
x_test = np.array([8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_test = np.array([8, 9, 10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(7))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 200, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # x,y test 값이 히든 레이어에 웨이트 값이 근사치 되게 도출되었는지 평가
print('loss : ', loss)# 로스 값을 프린트함
result = model.predict([11])# predict 값은 예측되어야 하는 값이므로 때에 따라 변경됨
print('11의 예측값 : ', result)

'''
loss :  4.000359510314411e-08
11의 예측값 :  [[10.999708]]
'''
