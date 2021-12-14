import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
              [5,6,7],[6,7,8],[7,8,9],[8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70]) 
# 80뽑기
print(x.shape, y.shape) # (13, 3) (13,)

#input_shape = (batch_size, timesteps, feature)
#input_shape = (행, 열, 몇개씩 자르는지!!!)
#!!reshape 바꿀때 데이터와 순서는 건들이면 안된다.
x = x.reshape(13,3,1,)


# #2. 모델구성
model = Sequential()
model.add(LSTM(80, activation = 'linear', input_shape = (3,1)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(36, activation = 'linear'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))
model.summary()
# #3. 컴파일, 훈련
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x, y ,epochs=500)

# #4. 평가, 예측
# model.evaluate(x, y)
# result = model.predict([[[50],[60],[70]]])
# print(result)
'''
Epoch 500/500
1/1 [==============================] - 0s 997us/step - loss: 8.9241e-04
1/1 [==============================] - 0s 132ms/step - loss: 8.9338e-04
[[80.05386]]
'''

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 80)                26240
_________________________________________________________________
dense (Dense)                (None, 64)                5184
_________________________________________________________________
dense_1 (Dense)              (None, 36)                2340
_________________________________________________________________
dense_2 (Dense)              (None, 13)                481
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 70
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 6
=================================================================
Total params: 34,321
Trainable params: 34,321
Non-trainable params: 0
_________________________________________________________________

'''
# RNN(순환 신경망)은 관련 정보와 그 정보를 사용하는 지점 사이 거리가 멀 경우 역전파시 
# 그래디언트가 점차 줄어 학습능력이 크게 저하되는 것으로 알려져 있습니다. 

# #LSTM params = 4 x (파라미터 아웃값 * (파라미터 아웃값 + 디멘션 값 + 1(바이어스)))
#                4    x (   80        x           (80     +     1     +     1))



# x4의 이유
# ft = σ(Wxh_fxt+Whh_fht−1+bh_f) =  ft는 ‘과거 정보를 잊기’를 위한 게이트입니다.
# it = σ(Wxh_ixt+Whh_iht−1+bh_i) = it⊙gt는 ‘현재 정보를 기억하기’ 위한 게이트입니다. 
# ot = σ(Wxh_oxt+Whh_oht−1+bh_o)
# gt = tanh(Wxh_gxt+Whh_ght−1+bh_g)

# ct = ft⊙ct−1+it⊙gt

# ht = ot⊙tanh(ct)


