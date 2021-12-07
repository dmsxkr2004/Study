import time
import pandas as pd #
import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
#from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api # 0값부터 연산
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
# print(x_train.shape,y_train.shape)
# print(x_test.shape,y_test.shape)
# print(y[0:10])
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 54))# linear= 기본값
model.add(Dense(50, activation = 'linear'))
model.add(Dropout(0.2))
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(7, activation = 'softmax'))
model.save("./_save/keras25_1_save_fetch.h5")
#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])  # metrics 몇개가 맞았는지 결과값을 보기위해 씀
                                                                                       
es = EarlyStopping(monitor='val_loss', patience=100, mode = 'auto')
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 500, validation_split = 0.2 , callbacks = [es], verbose = 1)
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)