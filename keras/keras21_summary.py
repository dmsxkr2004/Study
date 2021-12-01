# import tensorflow as tf
from tensorflow.keras.models import Sequential # 순차적(Sequential) 모델을 쓸것이다.
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))
# relu = y' = relu(wx + b)
# activation = linear, sigmoid, softmax

model.summary()


'''
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')#loss 값은 작으면 좋다 , loss 에 mse 값을 감축시키는 역할을 해줌(optimizer)

model.fit(x, y, epochs=3900, batch_size=1)#epochs = 훈련양

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)
result = model.predict([4])
print('4의 예측값 : ', result)
'''
'''
      ㅁ
ㅁ ㅁ ㅁ ㅁ ㅁ +b 10
  ㅁ ㅁ ㅁ ㅁ +b 24
    ㅁ ㅁ ㅁ +b  15
     ㅁ ㅁ   +b  8
      ㅁ     +b  3
위 형태의 레이어 구조일때 y = wx + b 형태에서
w1 x1 = 파라미터 형태로 레이어마다 들어가게되는데
레이어 마다 b값을 더해주는 구조로 연산이 된다.
'''