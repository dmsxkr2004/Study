import time
import numpy as np
from tensorflow.keras.models import Sequential
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

#2. 모델구성
model = Sequential()
model.add(Dense(70, input_dim = 10))
model.add(Dense(55))
model.add(Dense(40))
model.add(Dense(25,activation=relu))
model.add(Dense(10))
model.add(Dense(1))

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

'''
# 기본값
32/32 [==============================] - 0s 952us/step - loss: 3116.4329 - val_loss: 3587.8481
Epoch 00056: early stopping
걸린시간 :  2.43 초
3/3 [==============================] - 0s 501us/step - loss: 2160.0642
loss:  2160.064208984375
r2스코어 :  0.59460621380575
'''

'''
# MinMaxScaler
32/32 [==============================] - 0s 920us/step - loss: 3086.2988 - val_loss: 3480.0405
Epoch 00059: early stopping
걸린시간 :  2.513 초
3/3 [==============================] - 0s 568us/step - loss: 2057.0928
loss:  2057.0927734375
r2스코어 :  0.6139316466086311
'''
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
# StandardScaler
32/32 [==============================] - 0s 922us/step - loss: 3126.0398 - val_loss: 3628.4592
Epoch 00111: early stopping
걸린시간 :  4.134 초
3/3 [==============================] - 0s 726us/step - loss: 2184.4927
loss:  2184.49267578125
r2스코어 :  0.590021626019865
'''
'''
# relu를 사용한 StandardScaler
32/32 [==============================] - 0s 926us/step - loss: 2804.0125 - val_loss: 3454.8838
Epoch 00062: early stopping
걸린시간 :  2.719 초
3/3 [==============================] - 0s 864us/step - loss: 2156.4895
loss:  2156.489501953125
r2스코어 :  0.5952771843797104
'''
'''
# RobustScaler
32/32 [==============================] - 0s 913us/step - loss: 3083.8201 - val_loss: 3467.7991
Epoch 00105: early stopping
걸린시간 :  3.911 초
3/3 [==============================] - 0s 500us/step - loss: 2061.9941
loss:  2061.994140625
r2스코어 :  0.6130117862316242
'''

'''
# relu를 사용한 RobustScaler
32/32 [==============================] - 0s 979us/step - loss: 2848.1895 - val_loss: 3561.8218
Epoch 00080: early stopping
걸린시간 :  3.149 초
3/3 [==============================] - 0s 875us/step - loss: 2179.1765
loss:  2179.176513671875
r2스코어 :  0.5910194032936512
'''
'''
# MaxAbsScaler
32/32 [==============================] - 0s 968us/step - loss: 3255.2168 - val_loss: 3538.9224
Epoch 00054: early stopping
걸린시간 :  2.345 초
3/3 [==============================] - 0s 1ms/step - loss: 2176.8271
loss:  2176.8271484375
r2스코어 :  0.5914602648059508
'''
'''
# relu를 사용한 MaxAbsScaler
32/32 [==============================] - 0s 968us/step - loss: 3220.5059 - val_loss: 3303.7822
Epoch 00061: early stopping
걸린시간 :  2.566 초
3/3 [==============================] - 0s 630us/step - loss: 2102.3508
loss:  2102.350830078125
r2스코어 :  0.6054377482567814
'''