
import numpy as np
import time
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
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
print(x_train.shape) # (455, 30)
print(x_test.shape) # (114, 30)
print(y_train.shape) # (455,)
print(y_test.shape) # (114,)

x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)


#2. 모델구성
model = Sequential()
model.add(LSTM(70, activation = 'linear', input_shape = (30, 1)))# linear= 기본값
model.add(Dense(50, activation = 'linear'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###########################################################################
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# print(datetime)
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
model_path = "".join([filepath, 'cancer_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience = 50, mode = 'auto', verbose=1, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, validation_split = 0.2, callbacks = [es,mcp])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/cancer_1206_2327_0097-0.0201.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

results = model.predict(x_test[:11])
print(y_test[:11])
print(results)