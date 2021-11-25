from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time #시간에 대한 값들을 불러온다


#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

start = time.time()
model.fit(x, y, epochs=1000, batch_size = 1, verbose = 0)
end = time.time() - start
print('걸린시간 : ', end)
'''
verbose 0 = 결과값이 바로나옴
verbose 1 = 평소랑 같은 모든 결과창이 나옴
verbose 2 = loss값만나옴
verbose 3 = epoch 까지만 나옴
'''
'''
#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss: ', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)


'''
'''
loss:  0.38009995222091675
r2스코어 :  0.8099500481806899
'''