from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import time as time
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(70, input_dim = 13))
model.add(Dense(55))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

model.save("./_save/keras25_1_save_model.h5")
"""
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size = 1, validation_split = 0.2)
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
print("---------------------------------")
print(hist)
print("---------------------------------")
print(hist.history)
print("---------------------------------")
print(hist.history['loss'])
print("---------------------------------")
print(hist.history['val_loss'])
print("---------------------------------")

plt.figure(figsize = (9, 5))
plt.plot(hist.history['loss'], marker =',',c='red',label='loss')
plt.plot(hist.history['val_loss'], marker =',',c='blue',label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()
"""