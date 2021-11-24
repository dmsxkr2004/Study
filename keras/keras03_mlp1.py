import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])
x = np.transpose(x)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
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
y_predict = model.predict([[10, 1.3]])
print("[10, 1.3]의 예측값 : ",y_predict)
#열이 우선 행은 무시
#프레딕스 값은 디맨션(dim)값이랑 같아야한다.

# 출력값은 20 근사치 나옴
'''
loss :  0.07689565420150757
[10, 1.3]의 예측값 :  [[19.692553]]
'''
# https://rfriend.tistory.com/289 참고사이트 <ㅡ 트랜스포즈 설명 사이트
# 트랜스포즈 함수는 행과 열을 바꿔주는 함수