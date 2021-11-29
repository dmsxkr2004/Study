
# https://wikidocs.net/22647 참고 one hot encording
import time
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1 데이터
datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target
y = to_categorical(y)
# print(y.shape) #(150, 3)
# print(x.shape, y.shape) # (150, 4) (150,)
# print(y)
# print(np.unique(y))     #[0, 1, 2] # 다중분류
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)
print(x_train.shape,y_train.shape)#(120, 4) (120, 3)
print(x_test.shape,y_test.shape)#(30, 4) (30, 3)

#2. 모델구성
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 4))# linear= 기본값
model.add(Dense(50, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(3, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # binary_crossentropy 결과값이[0~1]로 한정될때 적합한 함수값 
                                                                                        # metrics 몇개가 맞았는지 결과값을 보기위해 씀
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
96/96 [==============================] - 0s 695us/step - loss: 0.0898 - accuracy: 0.9688 - val_loss: 0.0147 - val_accuracy: 1.0000
걸린시간 :  7.266 초
1/1 [==============================] - 0s 84ms/step - loss: 0.1398 - accuracy: 0.9000
loss:  0.13984128832817078
accuracy :  0.8999999761581421
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[2.00068834e-03 9.97571051e-01 4.28248488e-04]
 [1.06676649e-04 9.96195197e-01 3.69816320e-03]
 [1.31413748e-04 9.93274212e-01 6.59440784e-03]
 [9.99996305e-01 3.70121234e-06 1.07033496e-20]
 [6.63315412e-04 9.98682678e-01 6.53925352e-04]
 [1.05936360e-02 9.89016891e-01 3.89382039e-04]
 [9.99996305e-01 3.66873019e-06 2.74763292e-20]]
'''