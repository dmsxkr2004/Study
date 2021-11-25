from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7, shuffle=True, random_state=66)
#2. 모델구성
model = Sequential()
model.add(Dense(13, input_dim = 13))
model.add(Dense(25))
model.add(Dense(38))
model.add(Dense(53))
model.add(Dense(38))
model.add(Dense(21))
model.add(Dense(13))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=200, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
'''
loss:  16.81378746032715
r2스코어 :  0.796485395866257
'''






'''
print(x)
print(y)
print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)
'''
# e -03 = 0이 앞으로 3개 지수형 표현방식