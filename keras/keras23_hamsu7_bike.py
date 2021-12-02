import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. 데이터
path = "./_data/bike/"# .은 지금(현재 작업폴더) 스터디 폴더를 의미한다.
train =  pd.read_csv(path + "train.csv")# csv = 엑셀파일과 같다 , # pd.read_csv('경로/파일명.csv')
test_file =  pd.read_csv(path + "test.csv")
sampleSubmission_file = pd.read_csv(path + "sampleSubmission.csv")
#print(sampleSubmission_file.columns)
# print(train.shape) # (10886, 12)
# print(test.shape) # (6493, 9)
# print(sampleSubmission.shape) # (6493, 2)
#print(type(train))
#print(train.info())
#print(train.describe()) # std = 표준편차 min = 최소값 max = 최대값
#print(train.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')
#print(train.head())
#print(train.tail())

x = train.drop(['datetime','casual','registered','count'], axis = 1) # drop 리스트를 없애겠다. # axis
test_file = test_file.drop(['datetime'], axis = 1)
#print(x.columns)
#print(x.shape) # (10886, 8)
y = train['count']
#print(y)
print(y.shape) # (10886, )
y = np.log1p(y) # 로그변환
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size = 0.8, shuffle = True , random_state= 64)
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_test_file = scaler.transform(test_file)
#2. 모델구성
# model = Sequential()
# model.add(Dense(70, activation = 'linear', input_dim = 8))# linear= 기본값
# model.add(Dense(50))
# model.add(Dense(35, activation = 'relu'))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))
# model.summary()
input1 = Input(shape = (8,))
dense1 = Dense(70)(input1)
dense2 = Dense(50)(dense1)
dense3 = Dense(35, activation = 'relu')(dense2)
dense4 = Dense(20)(dense3)
dense5 = Dense(10)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs = input1, outputs = output1)
model.summary()
#3. 컴파일 훈련
model.compile(loss='mse', optimizer ='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto')
model.fit(x_train, y_train, epochs=1000, batch_size=50,
          verbose = 1, validation_split = 0.1, callbacks = [es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2스코어 : ', r2)
rmse = RMSE(y_test, y_pred)
print("RMSE : " , rmse)

plt.plot(y)
plt.show()