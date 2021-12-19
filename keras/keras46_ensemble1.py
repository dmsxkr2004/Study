import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
x1 = np.array([range(100), range(301, 401)]) # 삼성 저가, 종가
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)]) # 미국 국제선물 시가 고가 종가
x1 = np.transpose(x1)
x2 = np.transpose(x2)

y = np.array(range(1001, 1101)) # 타겟, 삼성전자 종가

print(x1.shape, x2.shape, y.shape) # (100, 2) (100, 3) (100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test ,y_train, y_test = train_test_split(
    x1, x2, y, train_size=0.7, random_state=66)

print(x1_train.shape, x2_train.shape) # (70, 2) (70, 3)
print(x1_test.shape, x2_test.shape) # (30, 2) (30, 3)
print(y_train.shape, y_test.shape) # (70,) (30,)

#2. 모델구성
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Input

# 2-1 모델 1

input1 = Input(shape = (2,))
dense1 = Dense(5, activation = 'relu', name = 'dense1') (input1)
dense2 = Dense(7, activation = 'relu', name = 'dense2') (dense1)
dense3 = Dense(7, activation = 'relu', name = 'dense3') (dense2)
output1 = Dense(7, activation = 'relu', name = 'output1') (dense3)

# 2-2 모델 2

input2 = Input(shape = (3,))
dense11 = Dense(10, activation = 'relu', name = 'dense11') (input2)
dense12 = Dense(10, activation = 'relu', name = 'dense12') (dense11)
dense13 = Dense(10, activation = 'relu', name = 'dense13') (dense12)
dense14 = Dense(10, activation = 'relu', name = 'dense14') (dense13)
output2 = Dense(5, activation = 'relu', name = 'output2') (dense14)

from tensorflow.keras.layers import concatenate, Concatenate
merge1 = Concatenate()([output1, output2])
merge2 = Dense(10, activation = 'relu')(merge1)
merge3 = Dense(7)(merge2)
last_output = Dense(1)(merge3)
model = Model(inputs = [input1, input2], outputs = last_output)

model.summary()
# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mae'])

model.fit([x1_train, x2_train], y_train, epochs = 200, batch_size = 1)


#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss[0])
print('loss : ', loss[1])
y_predict1 = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict1)
print('r2스코어 : ', r2)
# # # x = np.arange(20).reshape(2, 2, 5)
# # # print(x)
# # # y = np.arange(20, 30).reshape(2, 1, 5)
# # # print(y)
# # # tf.keras.layers.Concatenate(axis=1)([x, y])

# # # x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
# # # x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
# # # concatted = tf.keras.layers.Concatenate()([x1, x2])
# # # concatted.shape