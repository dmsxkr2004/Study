from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=49)

#2. 모델구성
model = Sequential()
model.add(Dense(60, input_dim = 10))
model.add(Dense(45))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=30, batch_size = 10, verbose = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


'''
print(x)
print(y)
print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)
'''
'''
loss:  1963.7900390625
r2스코어 :  0.6314423884030551
'''
# 트레인 사이즈 , 랜덤 스테이트 값에 따라서도 r2, loss값이 변경된다.
# 역피라미드 구조 레이어가 값이 대채적으로 잘 뜬다.

