# 실습!!!
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x = x.reshape(13,3,1) # 구성을 맞춰줘야 돌아간다.
#2. 모델구성
model = Sequential()
model.add(LSTM(10, input_shape = (3,1), return_sequences=True)) # (N, 3, 1) -> N, 10
model.add(LSTM(5, return_sequences=True))
model.add(LSTM(5, return_sequences=True))
model.add(LSTM(5, return_sequences=True))
model.add(LSTM(3))
model.add(Dense(5))
model.add(Dense(1))

# model.summary()

# #3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y ,epochs=1000)


#4. 평가, 예측
model.evaluate(x, y)
result = model.predict([[[50],[60],[70]]])
print(result)
