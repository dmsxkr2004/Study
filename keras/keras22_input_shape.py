import numpy as np
#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(701, 801)])
print(x.shape, y.shape) # (3, 100) (2, 100)
x = np.transpose(x) 
y = np.transpose(y) 
print(x.shape, y.shape) # (100, 3) (100, 1)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

model = Sequential()
#model.add(Dense(10, input_dim = 3))         #(100, 3) -> (N, 3) # input_dim = 다차원 데이터를 쓸때문제가됨 
model.add(Dense(10, input_shape = (3,)))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
model.summary()
'''
input dim

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 4
=================================================================
Total params: 117
Trainable params: 117
Non-trainable params: 0
_________________________________________________________________
'''
'''
input shape

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 4
=================================================================
Total params: 117
Trainable params: 117
Non-trainable params: 0
_________________________________________________________________
'''