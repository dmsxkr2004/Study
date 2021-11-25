## R2가 뭔지 찾기
'''
R2 지표도 회귀식의 성능을 평가하는 지표로 많이 사용하는 결정계수값으로 예측한 모델이 얼마나 실제 데이터를 설명하는지를 나타낸다
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 4, 3, 5, 7 ,9, 9, 8, 12, 13, 17, 12, 14, 21, 14, 11, 19, 23, 25])
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)
#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim = 1))
model.add(Dense(30))
model.add(Dense(70))
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')
# plt.show()

'''
loss:  7.547046184539795
r2스코어 :  0.4693485973599655
'''