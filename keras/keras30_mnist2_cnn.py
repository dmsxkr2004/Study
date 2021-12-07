import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(x_train.shape)

# print(np.unique(y_train, return_counts=True))
#2. 모델구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape = (28, 28, 1)))
model.add(Conv2D(5, (2,2), activation = 'relu')) # 7,7,5
model.add(Dropout(0.2))
model.add(Conv2D(7, (2,2), activation = 'relu')) # 7,7,5
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
###########################################################################
import datetime
date = datetime.datetime.now()
datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# print(datetime)
filepath = './_ModelCheckPoint/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
model_path = "".join([filepath, 'mnist_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=20, mode = 'auto', verbose=1)#, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=16, batch_size = 32, validation_split = 0.3, callbacks = [es,mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/mnist_1207_1843_0015-0.0745.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])

#평가지표 acc
#0.98 
# 발리데이션 스플릿으로 자르고
# 테스트로 
'''
loss:  0.07685080170631409
accuracy :  0.9815999865531921
'''