import time
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping patience(기다리는 횟수)
from tensorflow.python.keras.backend import relu
#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

#print(np.min(x), np.max(x))     # 0.0 711.0

#x = x/711.
#x = x/np.max(x)
x_train, x_test, y_train, y_test = train_test_split(x, y
                                    , shuffle=True ,train_size =0.8, random_state=49)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(70, input_dim = 10))
# model.add(Dense(55))
# model.add(Dense(40))
# model.add(Dense(25,activation=relu))
# model.add(Dense(10))
# model.add(Dense(1))
# model.save("./_save/keras25_1_save_diabetes.h5")
# # #3. 컴파일, 훈련
# # model.compile(loss='mse', optimizer='adam')
# # es = EarlyStopping(monitor='val_loss', patience=50, mode = 'min', verbose = 1)
# # start = time.time()
# # hist = model.fit(x_train, y_train, epochs=1000, batch_size = 10, 
# #                  validation_split = 0.1 , callbacks = [es])
# # end = time.time()- start

# # print("걸린시간 : ", round(end, 3), '초')
# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# ###########################################################################
# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# # print(datetime)
# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
# model_path = "".join([filepath, 'diabetes_', datetime, '_', filename])
#                 # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
# ############################################################################

# es = EarlyStopping(monitor= 'val_loss', patience=50, mode = 'auto', verbose=1, restore_best_weights = True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
#                       filepath = model_path)
# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size = 10, validation_split = 0.1, callbacks = [es,mcp])
# end = time.time()- start

# print("걸린시간 : ", round(end, 3), '초')
model = load_model('./_ModelCheckPoint/diabetes_1206_2318_0050-3306.4177.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

'''
#relu를 사용한 MinMaxScaler
32/32 [==============================] - 0s 919us/step - loss: 3262.1448 - val_loss: 3482.5105
Epoch 00060: early stopping
걸린시간 :  2.579 초
3/3 [==============================] - 0s 1ms/step - loss: 2053.4744
loss:  2053.474365234375
r2스코어 :  0.6146107239718315
'''
'''
loss:  2063.89501953125
r2스코어 :  0.6126549769727898
'''