# multi layer perceptron
'''
다층 퍼셉트론(Multilayer perceptron, MLP)은 퍼셉트론을 여러층 쌓은 순방향의 인공 신경망이다. 
입력층(input layer)과 은닉층(hidden layer)과 출력층(output layer)으로 구성된다. 
각 층에서는 활성함수를 통해 입력을 처리한다.
'''
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
x = np.transpose(x)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer ='adam')
model.fit(x, y, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss : ", loss)
y_predict = model.predict([[10, 1.3, 1]])
print("[10, 1.3, 1]의 예측값 : ",y_predict)

# [[10, 1.3, 1]]
'''
loss :  0.009113769046962261
[10, 1.3, 1]의 예측값 :  [[19.822159]]
'''