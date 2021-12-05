import time
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical # 값 백터수를 맞춰주는 api
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.python.keras.backend import relu

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
# print(x_train.shape,y_train.shape)#(120, 4) (120, 3)
# print(x_test.shape,y_test.shape)#(30, 4) (30, 3)
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(70, activation = 'linear', input_dim = 4))# linear= 기본값
model.add(Dense(55, activation = 'linear'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(25, activation = 'linear'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(3, activation = 'softmax'))
model.save("./_save/keras25_1_save_iris.h5")
#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # binary_crossentropy 결과값이[0~1]로 한정될때 적합한 함수값 
                                                                                        # metrics 몇개가 맞았는지 결과값을 보기위해 씀
es = EarlyStopping(monitor='val_loss', patience=50, mode = 'auto')
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 1, 
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
# 기본값
걸린시간 :  16.828 초
1/1 [==============================] - 0s 87ms/step - loss: 0.0840 - accuracy: 0.9667
loss:  0.08399715274572372
accuracy :  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[3.25390516e-04 9.99153137e-01 5.21442678e-04]
 [1.76822177e-05 9.93963540e-01 6.01876946e-03]
 [1.89970560e-05 9.88783956e-01 1.11969821e-02]
 [9.99997616e-01 2.43780096e-06 3.93769799e-21]
 [1.18784366e-04 9.98975873e-01 9.05455148e-04]
 [1.39578106e-03 9.98264015e-01 3.40267405e-04]
 [9.99996662e-01 3.39228086e-06 8.69997226e-21]]
'''
'''
# MinMaxScaler
걸린시간 :  9.994 초
1/1 [==============================] - 0s 90ms/step - loss: 0.0858 - accuracy: 1.0000
loss:  0.08579906076192856
accuracy :  1.0
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[8.5211461e-05 9.9942422e-01 4.9060903e-04]
 [4.7594514e-07 9.9137902e-01 8.6205145e-03]
 [1.7742609e-07 9.8384535e-01 1.6154392e-02]
 [1.0000000e+00 3.4303085e-16 1.9113976e-30]
 [3.1740921e-05 9.9889684e-01 1.0713068e-03]
 [2.3306452e-03 9.9752933e-01 1.4003729e-04]
 [1.0000000e+00 8.7421457e-15 2.9127900e-29]]
'''
'''
# relu를 사용한 MinMaxScaler
96/96 [==============================] - 0s 706us/step - loss: 0.0800 - accuracy: 0.9688 - val_loss: 0.5497 - val_accuracy: 0.8750
걸린시간 :  12.86 초
1/1 [==============================] - 0s 75ms/step - loss: 0.3696 - accuracy: 0.8667
loss:  0.3696041703224182
accuracy :  0.8666666746139526
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[5.23822855e-05 9.99947667e-01 1.23199424e-08]
 [7.96193126e-05 9.98560488e-01 1.35982793e-03]
 [1.47576984e-05 9.98827875e-01 1.15735026e-03]
 [1.00000000e+00 0.00000000e+00 0.00000000e+00]
 [1.08986060e-04 9.99889255e-01 1.77426762e-06]
 [8.06381255e-02 9.19361889e-01 1.12550455e-11]
 [1.00000000e+00 0.00000000e+00 0.00000000e+00]]
'''

'''
# StandardScaler
걸린시간 :  14.173 초
1/1 [==============================] - 0s 88ms/step - loss: 0.0925 - accuracy: 0.9667
loss:  0.09249864518642426
accuracy :  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[1.5047269e-05 9.9887413e-01 1.1108761e-03]
 [2.1498582e-07 9.8400533e-01 1.5994435e-02]
 [2.0884561e-07 9.7714841e-01 2.2851378e-02]
 [1.0000000e+00 3.9755097e-09 6.0996048e-22]
 [4.4067656e-06 9.9738580e-01 2.6097735e-03]
 [2.3145528e-04 9.9934930e-01 4.1926911e-04]
 [1.0000000e+00 2.7679162e-08 2.2175664e-21]]
'''
'''
#relu를 사용한 StandardScaler
96/96 [==============================] - 0s 697us/step - loss: 0.0383 - accuracy: 0.9792 - val_loss: 0.0670 - val_accuracy: 0.9583
걸린시간 :  7.783 초
1/1 [==============================] - 0s 84ms/step - loss: 0.1823 - accuracy: 0.9667
loss:  0.18228000402450562
accuracy :  0.9666666388511658
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[3.22252799e-06 9.99993563e-01 3.22412029e-06]
 [2.81659704e-05 9.88148868e-01 1.18229557e-02]
 [7.48751918e-05 9.84864712e-01 1.50603475e-02]
 [1.00000000e+00 6.39430219e-12 0.00000000e+00]
 [3.44150408e-06 9.99923110e-01 7.34232744e-05]
 [9.72876223e-05 9.99902606e-01 1.37217199e-07]
 [1.00000000e+00 4.06663342e-10 0.00000000e+00]]
'''
'''
# RobustScaler
걸린시간 :  9.773 초
1/1 [==============================] - 0s 88ms/step - loss: 0.0465 - accuracy: 1.0000
loss:  0.04650473594665527
accuracy :  1.0
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[3.3281464e-04 9.9944812e-01 2.1914388e-04]
 [5.7274497e-06 9.8820812e-01 1.1786142e-02]
 [1.2276440e-05 9.8294604e-01 1.7041640e-02]
 [9.9999416e-01 5.8243504e-06 2.4865825e-23]
 [5.9484566e-05 9.9909139e-01 8.4919517e-04]
 [5.7052625e-03 9.9425286e-01 4.1877152e-05]
 [9.9999440e-01 5.6439171e-06 8.1500163e-24]]
'''
'''
#relu를 사용한 RobustScaler
96/96 [==============================] - 0s 711us/step - loss: 0.0702 - accuracy: 0.9479 - val_loss: 0.0744 - val_accuracy: 0.9583
걸린시간 :  7.686 초
1/1 [==============================] - 0s 87ms/step - loss: 0.1055 - accuracy: 0.9333
loss:  0.10549184679985046
accuracy :  0.9333333373069763
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[1.4657647e-05 9.9998093e-01 4.3885507e-06]
 [9.1325732e-05 9.9893326e-01 9.7538449e-04]
 [2.0076170e-04 9.9644679e-01 3.3524870e-03]
 [1.0000000e+00 7.1984029e-20 0.0000000e+00]
 [2.5709009e-05 9.9991632e-01 5.7946480e-05]
 [8.2855567e-04 9.9917126e-01 2.2498304e-07]
 [1.0000000e+00 9.1106186e-18 0.0000000e+00]]
'''
'''
# MaxAbsScaler
걸린시간 :  16.23 초
1/1 [==============================] - 0s 85ms/step - loss: 0.0620 - accuracy: 1.0000
loss:  0.06202564761042595
accuracy :  1.0
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[2.2524783e-04 9.9839383e-01 1.3808464e-03]
 [1.3756719e-05 9.7765881e-01 2.2327431e-02]
 [1.3819102e-05 9.5848787e-01 4.1498378e-02]
 [9.9999118e-01 8.8240777e-06 1.7384772e-19]
 [8.7507207e-05 9.9704379e-01 2.8686982e-03]
 [9.4591483e-04 9.9864799e-01 4.0612757e-04]
 [9.9998856e-01 1.1488702e-05 1.6616997e-19]]
'''
'''
# relu를 사용한 MaxAbsScaler
96/96 [==============================] - 0s 675us/step - loss: 0.0599 - accuracy: 0.9792 - val_loss: 0.0314 - val_accuracy: 1.0000
걸린시간 :  12.048 초
1/1 [==============================] - 0s 82ms/step - loss: 0.1537 - accuracy: 0.9000
loss:  0.15370823442935944
accuracy :  0.8999999761581421
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
[[3.23267557e-07 9.99999523e-01 7.65135511e-08]
 [5.64466518e-09 9.99956846e-01 4.32087181e-05]
 [6.36324682e-09 9.99894500e-01 1.05519895e-04]
 [9.99999881e-01 1.37633990e-07 2.48123379e-38]
 [6.47269758e-08 9.99999404e-01 4.97566305e-07]
 [6.38623078e-06 9.99993563e-01 6.68907063e-09]
 [1.00000000e+00 4.48387674e-08 0.00000000e+00]]
'''