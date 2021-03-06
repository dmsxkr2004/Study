import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 8))# linear= 기본값
model.add(Dense(50))
model.add(Dense(35, activation = 'relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
# model.summary()
model.save("./_save/keras25_1_save_bike.h5")
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



# 로그 변환하기전엔 더하기 1을 해줌 이유는 로그 0 이 되어버리면 값을 정의할수없기때문에

################################### 제출용 제작############################################
results = model.predict(x_test_file)

sampleSubmission_file['count'] = results

print(sampleSubmission_file[:10])

sampleSubmission_file.to_csv(path + "submit6.csv", index = False)

'''
# 기본값
loss :  1.5395662784576416
r2스코어 :  0.22796990497314895
RMSE :  1.229973580148046
'''

'''
# MinMaxScaler
loss :  1.5111101865768433
r2스코어 :  0.24625883985977093
RMSE :  1.2292722438453276
'''

'''
# relu를 사용한 MinMaxScaler
loss :  1.4025722742080688
r2스코어 :  0.3003976818717775
RMSE :  1.1843023190011734
'''

'''
# StandardScaler
loss :  1.5365110635757446
r2스코어 :  0.23358896479107505
RMSE :  1.239560802317784
'''
'''
# relu를 사용한 StandardScaler
loss :  1.3943865299224854
r2스코어 :  0.30448060523001774
RMSE :  1.180841430518006
'''

'''
# RobustScaler
loss :  1.5238656997680664
r2스코어 :  0.23989655914879215
RMSE :  1.2344494457157265
'''
'''
# relu를 사용한 RobustScaler
loss :  1.3909422159194946
r2스코어 :  0.3061986436185332
RMSE :  1.179382100058195
'''
'''
# MaxAbsScaler
loss :  1.507941484451294
r2스코어 :  0.24783929839737873
RMSE :  1.2279827870551
'''
'''
# relu를 사용한 MaxAbsScaler
loss :  1.4285707473754883
r2스코어 :  0.2874297466616419
RMSE :  1.1952281250290642
'''
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 70)                630
_________________________________________________________________
dense_1 (Dense)              (None, 50)                3550
_________________________________________________________________
dense_2 (Dense)              (None, 35)                1785
_________________________________________________________________
dense_3 (Dense)              (None, 20)                720
_________________________________________________________________
dense_4 (Dense)              (None, 10)                210
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11
=================================================================
Total params: 6,906
Trainable params: 6,906
Non-trainable params: 0
_________________________________________________________________

activation 형태에때라서 변경사항 없음.
'''