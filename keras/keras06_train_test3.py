from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(100))# 개발자로서 숫자의 시작은 0부터 시작이다
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=70)
'''
print(x_test)     #[ 8 93  4  5 52 41  0 73 88 68]
print(y_test)     #[ 9 94  5  6 53 42  1 74 89 69]
'''
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
result = model.predict([100])# predict 값은 예측되어야 하는 값이므로 때에 따라 변경됨
print('100의 예측값 : ', result)



#7:3으로 배열 나누기
'''
x_train = np.random.choice(x, 70)
x_test = np.random.choice(x, 30)
y_train = np.random.choice(y, 70)
y_test = np.random.choice(y, 30)

print(x_train)
print(x_test)
print(y_train)
print(y_test)
'''
#https://rfriend.tistory.com/548 랜덤초이스 참고사이트