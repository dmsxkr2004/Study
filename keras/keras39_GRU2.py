import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70]) 
# 80뽑기
print(x.shape, y.shape) # (13, 3) (13,)

#input_shape = (batch_size, timesteps, feature)
#input_shape = (행, 열, 몇개씩 자르는지!!!)
#!!reshape 바꿀때 데이터와 순서는 건들이면 안된다.
x = x.reshape(13,3,1)


# #2. 모델구성
model = Sequential()
model.add(GRU(80, activation = 'linear', input_shape = (3,1)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(36, activation = 'linear'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))
# model.summary()

# #3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y ,epochs=10000)
start = time.time()
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')
#4. 평가, 예측
model.evaluate(x, y)
result = model.predict([[[50],[60],[70]]])
print(result)



