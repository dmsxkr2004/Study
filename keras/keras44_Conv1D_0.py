#시계열 데이터에서 보통 많이 사용한다. ex)주식
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional, Conv1D, Flatten # 대문자로 시작하면 통상 클래스 , 소문자로 시작하면 통상 함수

#1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])
y = np.array([4, 5, 6, 7]) 

print(x.shape, y.shape) # (4, 3) (4,)

#input_shape = (batch_size, timesteps, feature)
#input_shape = (행, 열, 몇개씩 자르는지!!!)
#!!reshape 바꿀때 데이터와 순서는 건들이면 안된다.

x = x.reshape(4, 3, 1)

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(50, input_shape = (3,1), return_sequences= True))
# model.add(Bidirectional(SimpleRNN(50)))
model.add(Conv1D(10, kernel_size= 2, input_shape = (3,1)))
model.add(Flatten())
model.add(Dense(40))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y ,epochs=500)

#4. 평가, 예측
model.evaluate(x, y)
result = model.predict([[[5],[6],[7]]])
print(result)
'''
[[8.025405]]
'''