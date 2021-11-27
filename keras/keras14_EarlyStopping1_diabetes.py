from sklearn.datasets import load_boston , load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import time as time
from tensorflow.python.keras.callbacks import History
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=49)

#2. 모델구성
model = Sequential()
model.add(Dense(70, input_dim = 10))
model.add(Dense(55))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping patience(기다리는 횟수)
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'min', verbose = 1)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 10, 
                 validation_split = 0.2 , callbacks = [es])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
'''
print("---------------------------------")
print(hist)
print("---------------------------------")
print(hist.history)
print("---------------------------------")
print(hist.history['loss'])
print("---------------------------------")
print(hist.history['val_loss'])
print("---------------------------------")
'''
plt.figure(figsize = (9, 5))
plt.plot(hist.history['loss'], marker =',',c='red',label='loss')
plt.plot(hist.history['val_loss'], marker =',',c='blue',label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show()
'''
29/29 [==============================] - 0s 1ms/step - loss: 3198.6184 - val_loss: 3530.5659
Epoch 00044: early stopping
걸린시간 :  2.012 초
3/3 [==============================] - 0s 750us/step - loss: 2270.0962
loss:  2270.09619140625
r2스코어 :  0.5739558402130422

자세한 내용은 보스턴 파일에 적어놓았음
'''