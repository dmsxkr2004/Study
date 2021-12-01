##############################################################
# 각각의 scaler의 특성과 정의 정리해놓을것
##############################################################
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.backend import relu
from tensorflow.python.keras.layers.core import Dropout # EarlyStopping patience(기다리는 횟수)
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

#print(np.min(x), np.max(x))     # 0.0 711.0

#x = x/711.
#x = x/np.max(x)
x_train, x_test, y_train, y_test = train_test_split(x, y
                                    ,train_size =0.7, random_state=66)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(70, input_dim = 13))
model.add(Dense(55))
model.add(Dense(40))
model.add(Dense(25, activation = relu))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'min', verbose = 1)
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, 
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
#그냥 결과값
283/283 [==============================] - 0s 574us/step - loss: 29.6942 - val_loss: 34.4657
Epoch 00244: early stopping
걸린시간 :  41.064 초
5/5 [==============================] - 0s 992us/step - loss: 19.0357
loss:  19.035663604736328
r2스코어 :  0.7695917365265258
'''
'''
# MinMaxScaler
283/283 [==============================] - 0s 598us/step - loss: 27.4960 - val_loss: 38.3810
Epoch 00058: early stopping
걸린시간 :  10.364 초
5/5 [==============================] - 0s 812us/step - loss: 20.9799
loss:  20.979862213134766
r2스코어 :  0.7460590958144051
'''
'''
# relu를 사용한 MinMaxScaler
283/283 [==============================] - 0s 577us/step - loss: 8.1544 - val_loss: 13.9863
Epoch 00274: early stopping
걸린시간 :  46.701 초
5/5 [==============================] - 0s 710us/step - loss: 9.5741
loss:  9.574063301086426
r2스코어 :  0.8841152492487321
'''
'''
# MaxAbsScaler
283/283 [==============================] - 0s 597us/step - loss: 26.2973 - val_loss: 33.4261
Epoch 00085: early stopping
걸린시간 :  14.739 초
5/5 [==============================] - 0s 747us/step - loss: 16.9458
loss:  16.945775985717773
r2스코어 :  0.7948878177088387
'''
'''
# relu를 사용한 MaxAbsScaler
283/283 [==============================] - 0s 605us/step - loss: 8.7999 - val_loss: 20.6038
Epoch 00255: early stopping
걸린시간 :  43.499 초
5/5 [==============================] - 0s 856us/step - loss: 9.5450
loss:  9.545010566711426
r2스코어 :  0.8844668903090628
'''
'''
# StandardScaler
283/283 [==============================] - 0s 579us/step - loss: 25.9200 - val_loss: 35.8133
Epoch 00085: early stopping
걸린시간 :  15.025 초
5/5 [==============================] - 0s 986us/step - loss: 19.2742
loss:  19.274152755737305
r2스코어 :  0.7667050539562521
'''
'''
# relu를 사용한 StandardScaler
283/283 [==============================] - 0s 589us/step - loss: 8.8458 - val_loss: 19.1998
Epoch 00111: early stopping
걸린시간 :  19.183 초
5/5 [==============================] - 0s 258us/step - loss: 11.6126
loss:  11.612607955932617
r2스코어 :  0.8594406299217788
'''
'''
# RobustScaler
283/283 [==============================] - 0s 577us/step - loss: 25.6738 - val_loss: 34.6252
Epoch 00058: early stopping
걸린시간 :  10.043 초
5/5 [==============================] - 0s 877us/step - loss: 17.2136
loss:  17.213577270507812
r2스코어 :  0.7916463038055531
'''
'''
#relu를 사용한 RobustScaler
283/283 [==============================] - 0s 585us/step - loss: 8.9520 - val_loss: 18.8930
Epoch 00196: early stopping
걸린시간 :  34.389 초
5/5 [==============================] - 0s 912us/step - loss: 11.2875
loss:  11.287548065185547
r2스코어 :  0.8633751742312398
'''