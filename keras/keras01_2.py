# import tensorflow as tf
from tensorflow.keras.models import Sequential # 순차적(Sequential) 모델을 쓸것이다.
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])
# 요 데이터를 훈련해서 최소의 loss를 만들어보자.

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')#loss 값은 작으면 좋다 , loss 에 mse 값을 감축시키는 역할을 해줌(optimizer)

model.fit(x, y, epochs=4000, batch_size=1)#epochs = 훈련양

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)
result = model.predict([6])
print('6의 예측값 : ', result)

'''
loss :  0.38020557165145874
6의 예측값 :  [[5.6748896]]
'''
#데이터가 불균형하기때문에 로스값이 점점 낮아지지 않고 0.38에서 머무르고 2차함수상 선을 그릴때 삐뚤하게 그리는 형상이 나옴