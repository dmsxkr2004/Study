import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

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
model.add(LSTM(80, activation = 'linear', input_shape = (3,1)))
model.add(Dense(64))
model.add(Dense(36, activation = 'tanh'))
model.add(Dense(13))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))
model.summary()
#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y ,epochs=500)

#4. 평가, 예측
model.evaluate(x, y)
result = model.predict([[[5],[6],[7]]])
print(result)
#LSTM params = 4 x (파라미터 아웃값 * (파라미터 아웃값 + 디멘션 값 + 1(바이어스)))
#              4         50        x           50     +     1     +     1


