from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Conv1D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import r2_score
import time
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)# (506, 13), (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=66)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape) # (404, 13)
print(y_train.shape) # (404,)
print(x_test.shape) # (102, 13)
print(y_test.shape) # (102,)

x_train = x_train.reshape(404, 13, 1)
x_test = x_test.reshape(102, 13, 1)

# #2. 모델구성

model = Sequential()
model.add(Conv1D(10, kernel_size= 2, input_shape = (13,1)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(35))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###########################################################################
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# print(datetime)
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
model_path = "".join([filepath, 'fashion_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=5, mode = 'auto', verbose=1, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.2, callbacks = [es,mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/fashion_1208_1144_0011-0.2975.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
