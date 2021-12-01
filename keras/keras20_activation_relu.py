# import tensorflow as tf
from tensorflow.keras.models import Sequential # 순차적(Sequential) 모델을 쓸것이다.
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.backend import relu
import numpy as np

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(2))
model.add(Dense(1))
# relu = y' = relu(wx + b)
# activation = linear, sigmoid, softmax , relu
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer ='adam')#loss 값은 작으면 좋다 , loss 에 mse 값을 감축시키는 역할을 해줌(optimizer)

model.fit(x, y, epochs=3900, batch_size=1)#epochs = 훈련양

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)
result = model.predict([4])
print('4의 예측값 : ', result)
'''
relu(rectified linear unit) 정리
특징:  0 이하의 값은 다음 레이어에 전달하지 않습니다. 0이상의 값은 그대로 출력합니다.
사용처: CNN을 학습시킬 때 많이 사용됩니다.
한계점: 한번 0 활성화 값을 다음 레이어에 전달하면 이후의 뉴런들의 출력값이 모두 0이 되는 현상이 발생합니다. 
이를 dying ReLU라 부릅니다. 이러한 한계점을 개선하기 위해 음수 출력 값을 소량이나마 다음 레이어에 전달하는 
방식으로 개선한 활성화 함수들이 등장합니다
'''