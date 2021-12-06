import time
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu

#1. 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target
y = to_categorical(y)
print(np.unique(y))
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)
# print(x_train.shape, y_train.shape)#(142, 13) (142, 3)
# print(x_test.shape, y_test.shape)#(36, 13) (36, 3)
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(70, activation = 'linear', input_dim = 13))# linear= 기본값
# model.add(Dense(50, activation = 'linear'))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dense(10, activation = 'linear'))
# model.add(Dense(3, activation = 'softmax'))
# model.save("./_save/keras25_1_save_wine.h5")
# # #3. 컴파일, 훈련
# # model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # categorical_crossentorpy 결과값이[0~3]이상일때 적합한 값 
# #                                                                                         # metrics 몇개가 맞았는지 결과값을 보기위해 씀 
# #                                                                                         # # 리스트[]는 다른 값이 더 들어갈수도 있음
# # es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto')
# # start = time.time()
# # hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, 
# #                  validation_split = 0.2 , callbacks = [es], verbose = 1)
# # end = time.time()- start

# # print("걸린시간 : ", round(end, 3), '초')
# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# ###########################################################################
# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# # print(datetime)
# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
# model_path = "".join([filepath, 'wine_', datetime, '_', filename])
#                 # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
# ############################################################################

# es = EarlyStopping(monitor= 'val_loss', patience=50, mode = 'auto', verbose=1, restore_best_weights = True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
#                       filepath = model_path)
# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, validation_split = 0.2, callbacks = [es,mcp])
# end = time.time()- start


# print("걸린시간 : ", round(end, 3), '초')
model = load_model('./_ModelCheckPoint/wine_1206_2338_0083-0.0598.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

'''
# relu를 사용한 MaxAbsScaler
113/113 [==============================] - 0s 684us/step - loss: 2.2922e-04 - accuracy: 1.0000 - val_loss: 0.0688 - val_accuracy: 0.9655
걸린시간 :  9.217 초
2/2 [==============================] - 0s 0s/step - loss: 0.0028 - accuracy: 1.0000
loss:  0.002753030275925994
accuracy :  1.0
'''
'''
loss:  0.014637116342782974
accuracy :  1.0
'''