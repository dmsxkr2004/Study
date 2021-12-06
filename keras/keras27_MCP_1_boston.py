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
from tensorflow.python.keras.saving.save import load_model
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

# #2. 모델구성
# model = Sequential()
# model.add(Dense(70, input_dim = 13))
# model.add(Dense(55))
# model.add(Dense(40))
# model.add(Dense(25, activation = relu))
# model.add(Dense(10))
# model.add(Dense(1))
# model.save("./_save/keras25_1_save_boston.h5")
# #3. 컴파일, 훈련
# '''
# model.compile(loss='mse', optimizer='adam')
# es = EarlyStopping(monitor='val_loss', patience=50, mode = 'min', verbose = 1)
# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, 
#                  validation_split = 0.2 , callbacks = [es])
# end = time.time()- start
# '''
# model.compile(loss='mse', optimizer='adam')
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# ###########################################################################
# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# # print(datetime)
# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
# model_path = "".join([filepath, 'boston_', datetime, '_', filename])
#                 # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
# ############################################################################

# es = EarlyStopping(monitor= 'val_loss', patience=10, mode = 'min', verbose=1, restore_best_weights = True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
#                       filepath = model_path)
# start = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.2, callbacks = [es,mcp])
# end = time.time()- start

# print("걸린시간 : ", round(end, 3), '초')
model = load_model('./_ModelCheckPoint/boston_1206_2312_0030-21.4241.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
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
loss:  11.911529541015625
r2스코어 :  0.8558224492064832
'''