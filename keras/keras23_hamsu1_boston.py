##############################################################
# 각각의 scaler의 특성과 정의 정리해놓을것
##############################################################
import time
import numpy as np
from tensorflow.keras.models import Sequential ,Model
from tensorflow.keras.layers import Dense ,Input
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
input1 = Input(shape = (13,))
dense1 = Dense(70)(input1)
dense2 = Dense(55)(dense1)
dense3 = Dense(40)(dense2)
dense4 =Dense(25, activation = relu)(dense3)
dense5 =Dense(10)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs = input1, outputs = output1)

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