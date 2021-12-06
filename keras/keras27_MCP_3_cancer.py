import numpy as np
import time
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu

#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
#print(datasets.feature_names)
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
#print(x.shape, y.shape) #(569, 30) (569,)
#print(y)
#print(np.unique(y))     #[0 1] # 이진분류 unique = 무슨 분류인지 알아보는 넘파이함수 값

# #2. 모델구성
# model = Sequential()
# model.add(Dense(70, activation = 'linear', input_dim = 30))# linear= 기본값
# model.add(Dense(50, activation = 'linear'))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dense(10, activation = 'linear'))
# model.add(Dense(1, activation = 'sigmoid'))# sigmoid 활성화 함수  0,1 로 한정해주는 함수
# model.save("./_save/keras25_1_save_cancer.h5")
# # #3. 컴파일, 훈련
# # model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # binary_crossentropy 결과값이[0~1]로 한정될때 적합한 함수값 
# #                                                                                         # metrics 몇개가 맞았는지 결과값을 보기위해 씀
# # es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto')
# # start = time.time()
# # hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, 
# #                  validation_split = 0.2 , callbacks = [es], verbose = 1)
# # end = time.time()- start

# # print("걸린시간 : ", round(end, 3), '초')
# #3. 컴파일, 훈련
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# ###########################################################################
# import datetime
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# # print(datetime)
# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
# model_path = "".join([filepath, 'cancer_', datetime, '_', filename])
#                 # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
# ############################################################################

# es = EarlyStopping(monitor= 'val_loss', patience = 50, mode = 'auto', verbose=1, restore_best_weights = True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
#                       filepath = model_path)
# start = time.time()
# hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, validation_split = 0.2, callbacks = [es,mcp])
# end = time.time()- start

# print("걸린시간 : ", round(end, 3), '초')
model = load_model('./_ModelCheckPoint/cancer_1206_2327_0097-0.0201.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

results = model.predict(x_test[:11])
print(y_test[:11])
print(results)

'''
#relu를 사용한 MaxAbsScaler
364/364 [==============================] - 0s 574us/step - loss: 0.0462 - accuracy: 0.9863 - val_loss: 0.0723 - val_accuracy: 0.9560
걸린시간 :  37.682 초
4/4 [==============================] - 0s 655us/step - loss: 0.3358 - accuracy: 0.9474
loss:  [0.33579370379447937, 0.9473684430122375]
'''
'''
4/4 [==============================] - 0s 665us/step - loss: 0.1875 - accuracy: 0.9737
loss:  [0.1874733567237854, 0.9736841917037964]
'''