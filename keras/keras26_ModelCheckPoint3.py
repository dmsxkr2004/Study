from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import time as time
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(70, input_dim = 13))
model.add(Dense(55))
model.add(Dense(40))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor= 'val_loss', patience=10, mode = 'min', verbose=1, restore_best_weights = False)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = './_ModelCheckPoint/keras26_3_MCP.hdf5')
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.2, callbacks = [es,mcp])
end = time.time()- start


print("걸린시간 : ", round(end, 3), '초')
model.save("./_save/keras26_3_save_model.h5")
# model = load_model('./_ModelCheckPoint/keras26_1_MCP.hdf5')
# model = load_model('./_save/keras26_1_save_weights.h5')

#4. 평가, 예측

print("==============================1. 기본출력 ===========================")
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
print("==============================2. load_model출력 ===========================")
model2 = load_model('./_save/keras26_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss: ', loss2)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)
print("==============================3. ModelCheckPoint 출력 ===========================")
model3 = load_model('./_ModelCheckPoint/keras26_3_MCP.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)