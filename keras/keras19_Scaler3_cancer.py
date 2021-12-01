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
model.add(Dense(50, activation = 'linear'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(1, activation = 'sigmoid'))# sigmoid 활성화 함수  0,1 로 한정해주는 함수

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

'''
# 기본값
Epoch 301/1000
364/364 [==============================] - 0s 607us/step - loss: 0.1211 - accuracy: 0.9533 - val_loss: 0.1727 - val_accuracy: 0.9231
걸린시간 :  65.372 초
4/4 [==============================] - 0s 997us/step - loss: 0.2120 - accuracy: 0.9298
loss:  [0.21200057864189148, 0.9298245906829834]
[1 1 1 1 1 0 0 1 1 1 0]
[[9.4568360e-01]
 [9.7557604e-01]
 [9.9267280e-01]
 [9.7197890e-01]
 [8.1307125e-01]
 [9.9365115e-03]
 [4.7213439e-15]
 [3.0479431e-02]
 [9.8433977e-01]
 [9.9322689e-01]
 [1.3942002e-06]]
'''
'''
# MinMaxScaler
Epoch 101/1000
364/364 [==============================] - 0s 560us/step - loss: 0.0585 - accuracy: 0.9780 - val_loss: 0.0357 - val_accuracy: 0.9780
걸린시간 :  22.029 초
4/4 [==============================] - 0s 1ms/step - loss: 0.1371 - accuracy: 0.9649
loss:  [0.13710394501686096, 0.9649122953414917]
[1 1 1 1 1 0 0 1 1 1 0]
[[9.9725354e-01]
 [9.9981296e-01]
 [9.9969876e-01]
 [9.9992740e-01]
 [9.7499532e-01]
 [1.1484981e-02]
 [1.1708943e-12]
 [9.9999851e-01]
 [9.9966526e-01]
 [9.9981844e-01]
 [6.3148644e-07]]
'''
'''
# relu를 사용한 MinMaxScaler
364/364 [==============================] - 0s 600us/step - loss: 0.0326 - accuracy: 0.9918 - val_loss: 0.0914 - val_accuracy: 0.9560
걸린시간 :  17.064 초
4/4 [==============================] - 0s 1ms/step - loss: 0.1647 - accuracy: 0.9561
loss:  [0.16474972665309906, 0.9561403393745422]
[1 1 1 1 1 0 0 1 1 1 0]
[[9.9096990e-01]
 [9.4958365e-01]
 [9.9995017e-01]
 [9.9973750e-01]
 [9.9941540e-01]
 [1.1195350e-05]
 [2.5726555e-30]
 [9.3351555e-01]
 [9.5384765e-01]
 [9.9995244e-01]
 [5.2095309e-15]]
'''
'''
# StandardScaler
Epoch 94/1000
364/364 [==============================] - 0s 664us/step - loss: 0.0506 - accuracy: 0.9780 - val_loss: 0.0440 - val_accuracy: 0.9780
걸린시간 :  20.527 초
4/4 [==============================] - 0s 1ms/step - loss: 0.1538 - accuracy: 0.9737
loss:  [0.15379919111728668, 0.9736841917037964]
[1 1 1 1 1 0 0 1 1 1 0]
[[9.9975926e-01]
 [9.9999833e-01]
 [9.9998927e-01]
 [9.9999899e-01]
 [9.9768519e-01]
 [5.3646232e-05]
 [6.9171153e-21]
 [1.0000000e+00]
 [9.9971086e-01]
 [9.9999750e-01]
 [6.8754273e-11]]
'''
'''
# relu를 사용한 StandardScaler
364/364 [==============================] - 0s 577us/step - loss: 3.1817e-07 - accuracy: 1.0000 - val_loss: 0.1063 - val_accuracy: 0.9780
걸린시간 :  14.015 초
4/4 [==============================] - 0s 0s/step - loss: 0.8512 - accuracy: 0.9561
loss:  [0.8512303829193115, 0.9561403393745422]
[1 1 1 1 1 0 0 1 1 1 0]
[[1.0000000e+00]
 [1.0000000e+00]
 [1.0000000e+00]
 [1.0000000e+00]
 [1.0000000e+00]
 [3.3488123e-17]
 [0.0000000e+00]
 [1.0000000e+00]
 [1.0000000e+00]
 [1.0000000e+00]
 [0.0000000e+00]]
'''
'''
# RobustScaler
Epoch 61/1000
364/364 [==============================] - 0s 606us/step - loss: 0.0474 - accuracy: 0.9808 - val_loss: 0.0399 - val_accuracy: 0.9780
걸린시간 :  13.798 초
4/4 [==============================] - 0s 1ms/step - loss: 0.1064 - accuracy: 0.9737
loss:  [0.10640352964401245, 0.9736841917037964]
[1 1 1 1 1 0 0 1 1 1 0]
[[9.9663281e-01]
 [9.9993956e-01]
 [9.9970698e-01]
 [9.9995875e-01]
 [9.6809864e-01]
 [2.4169683e-04]
 [2.0960669e-15]
 [9.9999976e-01]
 [9.9461991e-01]
 [9.9988770e-01]
 [3.8768988e-08]]
'''
'''
# relu를 사용한 RobustScaler
364/364 [==============================] - 0s 581us/step - loss: 3.0138e-07 - accuracy: 1.0000 - val_loss: 0.1463 - val_accuracy: 0.9670
걸린시간 :  14.633 초
4/4 [==============================] - 0s 0s/step - loss: 0.8606 - accuracy: 0.9386
loss:  [0.860579252243042, 0.9385964870452881]
[1 1 1 1 1 0 0 1 1 1 0]
[[1.00000e+00]
 [1.00000e+00]
 [1.00000e+00]
 [1.00000e+00]
 [1.00000e+00]
 [3.21504e-21]
 [0.00000e+00]
 [1.00000e+00]
 [1.00000e+00]
 [1.00000e+00]
 [0.00000e+00]]
'''
'''
# MaxAbsScaler
Epoch 83/1000
364/364 [==============================] - 0s 589us/step - loss: 0.0712 - accuracy: 0.9753 - val_loss: 0.0404 - val_accuracy: 0.9890
걸린시간 :  18.444 초
4/4 [==============================] - 0s 873us/step - loss: 0.0774 - accuracy: 0.9737
loss:  [0.07742246985435486, 0.9736841917037964]
[1 1 1 1 1 0 0 1 1 1 0]
[[9.8660529e-01]
 [9.9745977e-01]
 [9.9910796e-01]
 [9.9950010e-01]
 [9.4222689e-01]
 [1.5943587e-02]
 [3.7826790e-12]
 [9.9980223e-01]
 [9.9877745e-01]
 [9.9842715e-01]
 [2.9501274e-07]]
'''
'''
#relu를 사용한 MaxAbsScaler
364/364 [==============================] - 0s 574us/step - loss: 0.0462 - accuracy: 0.9863 - val_loss: 0.0723 - val_accuracy: 0.9560
걸린시간 :  37.682 초
4/4 [==============================] - 0s 655us/step - loss: 0.3358 - accuracy: 0.9474
loss:  [0.33579370379447937, 0.9473684430122375]
[1 1 1 1 1 0 0 1 1 1 0]
[[9.9996662e-01]
 [9.9926013e-01]
 [9.9999946e-01]
 [9.9999774e-01]
 [9.9953288e-01]
 [4.2544311e-01]
 [3.0596712e-18]
 [9.9999976e-01]
 [9.1679126e-01]
 [9.9999899e-01]
 [4.8062305e-09]]
'''