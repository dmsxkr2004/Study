import time
import pandas as pd #
import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
#from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api # 0값부터 연산
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu
#from sklearn.preprocessing import OneHotEncoder # 0~7까지 출력

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
#y = to_categorical(y) 
'''
ohe = OneHotEncoder(sparse=False) # 1부터 끝값까지 출력한다.
y = ohe.fit_transform(y.reshape(-1, 1))
'''
y = pd.get_dummies(y) # 0데이터값을 빼고 그 다음부터 연산
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape,y_train.shape) # (464809, 54) (464809, 7)
print(x_test.shape,y_test.shape) # (116203, 54) (116203, 7)
# print(y[0:10])
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(464809, 54, 1)
x_test = x_test.reshape(116203, 54, 1)
# #2. 모델구성
model = Sequential()
model.add(LSTM(70, activation = 'linear', input_shape = (54, 1)))# linear= 기본값
model.add(Dense(50, activation = 'linear'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(7, activation = 'softmax'))
# model.save("./_save/keras25_1_save_fetch.h5")

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###########################################################################
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# print(datetime)
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
model_path = "".join([filepath, 'fetch_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=10, mode = 'auto', verbose=1, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size = 500, validation_split = 0.2, callbacks = [es,mcp])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/fetch_1206_2341_0725-0.4223.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)