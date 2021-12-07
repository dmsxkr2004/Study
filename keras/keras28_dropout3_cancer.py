import numpy as np
import time
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu
from tensorflow.python.keras.layers.core import Dropout

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

#2. 모델구성
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 30))# linear= 기본값
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'linear'))
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(1, activation = 'sigmoid'))# sigmoid 활성화 함수  0,1 로 한정해주는 함수
model.save("./_save/keras25_1_save_cancer.h5")
#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # binary_crossentropy 결과값이[0~1]로 한정될때 적합한 함수값 
                                                                                        # metrics 몇개가 맞았는지 결과값을 보기위해 씀
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto')
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, 
                 validation_split = 0.2 , callbacks = [es], verbose = 1)
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

results = model.predict(x_test[:11])
print(y_test[:11])
print(results)
