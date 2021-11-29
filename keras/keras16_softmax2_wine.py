#다중분류(default = y 값을 one hot encording 해준다.) 
import time
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

datasets = load_wine()

x = datasets.data
y = datasets.target
y = to_categorical(y)
print(np.unique(y))
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape, y_train.shape)#(142, 13) (142, 3)
print(x_test.shape, y_test.shape)#(36, 13) (36, 3)

#2. 모델구성
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 13))# linear= 기본값
model.add(Dense(50, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(3, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # categorical_crossentorpy 결과값이[0~3]이상일때 적합한 값 
                                                                                        # metrics 몇개가 맞았는지 결과값을 보기위해 씀 
                                                                                        # # 리스트[]는 다른 값이 더 들어갈수도 있음
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto')
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size = 1, 
                 validation_split = 0.2 , callbacks = [es], verbose = 1)
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
113/113 [==============================] - 0s 643us/step - loss: 0.3068 - accuracy: 0.9027 - val_loss: 0.3935 - val_accuracy: 0.8621
걸린시간 :  8.455 초
2/2 [==============================] - 0s 0s/step - loss: 0.2060 - accuracy: 0.9444
loss:  0.20599092543125153
accuracy :  0.9444444179534912
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[2.7288002e-04 3.8425557e-02 9.6130157e-01]
 [1.7258385e-01 8.0731565e-01 2.0100480e-02]
 [4.9251712e-06 9.9996352e-01 3.1531243e-05]
 [9.9895132e-01 2.8068511e-04 7.6792721e-04]
 [9.9804667e-05 9.9912673e-01 7.7344483e-04]
 [6.8084432e-06 9.9989641e-01 9.6785909e-05]
 [9.5041633e-02 5.0437418e-03 8.9991462e-01]]
 '''