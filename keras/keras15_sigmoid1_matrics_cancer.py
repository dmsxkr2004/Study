#이진분류(default = sigmoid) , 다중분류(default = y 값을 one hot encording 해준다.) , 회귀분류(default = linear) 세가지 유형이 있는데 지금 하는 유형은 이진분류 형이다.
import numpy as np
import time
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_breast_cancer()
#print(datasets)
#print(datasets.DESCR)
#print(datasets.feature_names)
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)
#print(x.shape, y.shape) #(569, 30) (569,)
#print(y)
#print(np.unique(y))     #[0 1] # 이진분류 unique = 무슨 분류인지 알아보는 넘파이함수 값

#2. 모델구성
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 30))# linear= 기본값
model.add(Dense(50, activation = 'linear'))
model.add(Dense(30, activation = 'linear'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(1, activation = 'sigmoid'))# sigmoid 활성화 함수  0,1 로 한정해주는 함수

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # binary_crossentropy 결과값이[0~1]로 한정될때 적합한 함수값 
                                                                                        # metrics 몇개가 맞았는지 결과값을 보기위해 씀
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto')
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size = 1, 
                 validation_split = 0.2 , callbacks = [es], verbose = 1)
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)

results = model.predict(x_test[:11])
print(y_test[:11])
print(results)
# y_predict = model.predict(x_test)

# print('predict값 : ', y_predict)
'''
364/364 [==============================] - 0s 587us/step - loss: 0.1633 - accuracy: 0.9313 - val_loss: 0.0877 - val_accuracy: 0.9560
걸린시간 :  21.938 초
4/4 [==============================] - 0s 0s/step - loss: 0.1883 - accuracy: 0.9211
 loss:  [0.18832698464393616, 0.9210526347160339]
[1 1 1 1 1 0 0 1 1 1 0]
[[9.71834362e-01]
 [9.92785037e-01]
 [9.93371487e-01]
 [9.79388535e-01]
 [9.27746773e-01]
 [1.01772904e-01]
 [2.19984811e-13]
 [2.37003177e-01]
 [9.90237653e-01]
 [9.95357275e-01]
 [2.63124704e-04]]
'''

'''
def aaa (x):
        if x > 0.5:
        a = 1
        else a = 0
        return a
'''


