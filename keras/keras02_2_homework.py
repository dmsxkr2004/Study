# import tensorflow as tf
from tensorflow.keras.models import Sequential # 순차적(Sequential) 모델을 쓸것이다.
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))# 이 줄이 하나의 레이어
model.add(Dense(45))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
# 하이퍼 파라미터 튜닝

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')#loss 값은 작으면 좋다 , loss 에 mse 값을 감축시키는 역할을 해줌(optimizer)

model.fit(x, y, epochs=30, batch_size=1)#epochs = 훈련양

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)
result = model.predict([4])
print('4의 예측값 : ', result)
'''
epochs = 30을 주고 4의 예측값을 하이퍼 파라미터 레이어를 튜닝해서 값을 출력하기 
'''