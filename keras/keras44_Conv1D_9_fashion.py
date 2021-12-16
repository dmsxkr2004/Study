from tensorflow.keras.datasets import fashion_mnist
import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time

#1.데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# scaler = StandardScaler()

# n = x_train.shape[0]# 이미지갯수 50000
# x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
# x_train = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1
# # x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

# m = x_test.shape[0]
# x_test = scaler.transform(x_test.reshape(m,-1))#.reshape(x_test.shape)

#2. 모델구성
model = Sequential()
# model.add(Dense(64, input_shape = (28*28, )))
model.add(Conv1D(10, kernel_size= 2, input_shape = (28,28)))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(35, activation = 'relu'))
model.add(Dense(20))
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
model_path = "".join([filepath, 'fashion_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=5, mode = 'auto', verbose=1, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=16, batch_size = 32, validation_split = 0.2, callbacks = [es,mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/fashion_1208_1144_0011-0.2975.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])
'''
loss:  0.34458884596824646
accuracy :  0.8751000165939331
'''
'''
loss:  0.4457622468471527
accuracy :  0.8349999785423279
'''
'''
loss:  0.6096596717834473
accuracy :  0.7835999727249146
'''