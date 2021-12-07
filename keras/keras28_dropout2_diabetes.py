import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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

#2. 모델구성
model = Sequential()
model.add(Dense(70, input_dim = 10))
model.add(Dense(55))
model.add(Dropout(0.2))
model.add(Dense(40))
model.add(Dense(25,activation=relu))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))
model.save("./_save/keras25_1_save_diabetes.h5")
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'min', verbose = 1)
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 10, 
                 validation_split = 0.1 , callbacks = [es])
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
