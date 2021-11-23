# import tensorflow as tf
from tensorflow.keras.models import Sequential # 순차적(Sequential) 모델을 쓸것이다.
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')#loss 값은 작으면 좋다 , loss 에 mse 값을 감축시키는 역할을 해줌(optimizer)

model.fit(x, y, epochs=3900, batch_size=1)#epochs = 훈련양

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)
result = model.predict([4])
print('4의 예측값 : ', result)

'''
loss :  0.0015667766565456986
4의 예측값 :  [[3.9181228]]
'''
#4000 가까이 실행을 돌리면 최적의 값을 낼수 있음
#많은 데이터를 비교해본다고 정확하진 않음
#로스값이 작을수록 정확하다