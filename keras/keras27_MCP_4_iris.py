import time
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu

#1 데이터
datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
y = to_categorical(y)
# print(y.shape) #(150, 3)
# print(x.shape, y.shape) # (150, 4) (150,)
# print(y)
# print(np.unique(y))     #[0, 1, 2] # 다중분류
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)
# print(x_train.shape,y_train.shape)#(120, 4) (120, 3)
# print(x_test.shape,y_test.shape)#(30, 4) (30, 3)
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(70, activation = 'linear', input_dim = 4))# linear= 기본값
# model.add(Dense(55, activation = 'linear'))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(25, activation = 'linear'))
# model.add(Dense(10, activation = 'linear'))
# model.add(Dense(3, activation = 'softmax'))
# model.save("./_save/keras25_1_save_iris.h5")
# # #3. 컴파일, 훈련
# # model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # binary_crossentropy 결과값이[0~1]로 한정될때 적합한 함수값 
# #                                                                                         # metrics 몇개가 맞았는지 결과값을 보기위해 씀
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
# model_path = "".join([filepath, 'iris_', datetime, '_', filename])
#                 # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
# ############################################################################

# es = EarlyStopping(monitor= 'val_loss', patience=50, mode = 'auto', verbose=1, restore_best_weights = True)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
#                       filepath = model_path)
# start = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.2, callbacks = [es,mcp])
# end = time.time()- start


# print("걸린시간 : ", round(end, 3), '초')
model = load_model('./_ModelCheckPoint/iris_1206_2333_0050-0.0031.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

'''
# MinMaxScaler
걸린시간 :  9.994 초
1/1 [==============================] - 0s 90ms/step - loss: 0.0858 - accuracy: 1.0000
loss:  0.08579906076192856
accuracy :  1.0
'''
'''
loss:  0.07469351589679718
accuracy :  0.9666666388511658
'''