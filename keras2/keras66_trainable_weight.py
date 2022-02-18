import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
model.summary()

print(model.weights)
print("====================================")
print(model.trainable_weights)
print("====================================")

print(len(model.weights)) # 6
print(len(model.trainable_weights)) # 6

# model.trainable = False

print(len(model.weights)) # 6
print(len(model.trainable_weights)) # 6

model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, batch_size = 1, epochs= 100)

#4. 평가, 예측
