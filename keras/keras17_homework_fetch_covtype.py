import time
import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 54))# linear= 기본값
model.add(Dense(50, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(8, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])  # metrics 몇개가 맞았는지 결과값을 보기위해 씀
                                                                                       
es = EarlyStopping(monitor='val_loss', patience=20, mode = 'auto')
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size = 54, validation_split = 0.2 , callbacks = [es], verbose = 1)
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])

results = model.predict(x_test[:7])
print(y_test[:7])
print(results)

'''
6887/6887 [==============================] - 4s 613us/step - loss: 0.6557 - accuracy: 0.7152 - val_loss: 0.6511 - val_accuracy: 0.7196
걸린시간 :  280.177 초
3632/3632 [==============================] - 1s 380us/step - loss: 0.6512 - accuracy: 0.7200
loss:  0.6512410640716553
accuracy :  0.7200244665145874
[[0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]]
[[4.4781895e-13 7.4952549e-01 2.2385982e-01 2.1126117e-07 1.9213609e-12
  3.0120770e-03 1.5551593e-07 2.3602324e-02]
 [1.0985118e-09 9.3759894e-02 9.0364629e-01 2.7151030e-04 1.5604428e-05
  1.9816277e-03 2.9555251e-04 2.9509951e-05]
 [6.0420543e-13 8.1721109e-01 1.6534135e-01 5.8883171e-07 2.1973567e-11
  1.9002964e-03 5.3934846e-06 1.5541326e-02]
 [3.4911183e-09 1.0451855e-01 8.8227618e-01 6.0921483e-04 1.2990119e-04
  1.1241197e-02 9.9892879e-04 2.2596522e-04]
 [5.7209129e-12 4.6589664e-01 5.1567745e-01 3.9640523e-05 9.6943314e-11
  1.0674973e-02 3.9230104e-05 7.6720458e-03]
 [3.9742431e-13 1.8019772e-01 7.7716565e-01 2.2048436e-05 3.2487277e-10
  4.2507198e-02 7.6138342e-05 3.1192772e-05]
 [4.0364659e-10 2.7299038e-01 7.2431803e-01 7.3583309e-05 8.1914372e-08
  1.2266131e-03 8.2933926e-05 1.3084153e-03]]
'''
'''
6일차 AI 연산 분류 정리
------------------------------------------------------------------------
회귀분류(default = linear) 보통의 값 
y = wx + b 값으로 파라미터 형식으로 데이터를 한방향으로 보냄
직선으로 함수를 그어서 데이터값이 직선과 맞닫게 로직을 구현함
outliner =  쓰레기 데이터
ex) 나이를 통계로 든다면 1000이나 나올수없는수들을 의미함 
val_loss값과 loss값으로 데이터를 판별함
------------------------------------------------------------------------
이진분류(default = sigmoid) [0,1] 로 구성되어있는 데이터들 아웃풋 값에 적용
아웃풋이 0~1 사이에 포함됨
binary_crossentropy 로 loss값을 나타냄
------------------------------------------------------------------------
다중분류(default = softmax) 
categorical_crossentropy 로 loss값을 나타냄
# https://wikidocs.net/22647 참고 one hot encording
3개 이상의 값을 나오는 값 따라 적용함
ex) 0.5 / 0.3 / 0.2  = 1 이런식으로 3개의 아웃풋 값이 나오면 합이 1인 형태로 나오고 0.5의 개 데이터값으로 출력함
    개  고양이  표범
'''
