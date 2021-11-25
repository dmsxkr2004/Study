from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=200, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)
'''
loss:  0.38009995222091675
r2스코어 :  0.8099500481806899
'''