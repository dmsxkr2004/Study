import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6]])
y = np.array([4, 5, 6, 7]) 

print(x.shape, y.shape) # (4, 3) (4,)



#!!reshape 바꿀때 데이터와 순서는 건들이면 안된다.
x = x.reshape(4, 3, 1)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(50, input_shape = (3,1)))
model.add(Dense(40))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()
# #3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x, y ,epochs=500)

# #4. 평가, 예측
# model.evaluate(x, y)
# result = model.predict([[[5],[6],[7]]])
# print(result)

#input_shape = (batch_size, timesteps, feature)
#input_shape = (행, 열, 몇개씩 자르는지!!!)

# RNN(순환 신경망)은 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀 경우 역전파시 
# 그래디언트가 점차 줄어 학습능력이 크게 저하되는 것으로 알려져 있습니다. 

#params = 파라미터 아웃값 * (파라미터 아웃값 + 디멘션 값 + 1(바이어스))
#              50        x           50     +     1     +     1
