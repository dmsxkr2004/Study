import numpy as np
#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(711, 811), range(101, 201)])
print(x.shape, y.shape) # (3, 100) (2, 100)
x = np.transpose(x) 
y = np.transpose(y) 
print(x.shape, y.shape) # (100, 3) (100, 2)

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape = (3,))
dense1 = Dense(10)(input1)
dense2 = Dense(5, activation = 'relu')(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)

# model = Sequential()
# #model.add(Dense(10, input_dim = 3))         #(100, 3) -> (N, 3) # input_dim = 다차원 데이터를 쓸때문제가됨 
# model.add(Dense(10, input_shape = (3,)))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))
model.summary()