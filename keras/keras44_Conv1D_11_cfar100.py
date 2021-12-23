from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D
from tensorflow.keras.utils import to_categorical
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
#1.데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.reshape(50000, 32, 96)
x_test = x_test.reshape(10000, 32, 96)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, x_test.shape) #(50000, 100) (10000, 100)

'''
scaler = StandardScaler()
n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1
x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)
'''
# #2. 모델구성
model = Sequential()
model.add(Conv1D(300, kernel_size= 2, input_shape = (32,96)))
model.add(Flatten())
model.add(Dense(200, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))

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
model_path = "".join([filepath, 'cifar100_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=20, mode = 'auto', verbose=1, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=32, batch_size = 32, validation_split = 0.1, callbacks = [es,mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/cifar100_1208_2333_0020-3.3161.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])
# plt.imshow(x_train[0])
# plt.show()
'''# Conv2D
loss:  3.2576427459716797
accuracy :  0.22460000216960907
'''
'''# LSTM
loss:  4.612941265106201
accuracy :  0.009999999776482582
'''
'''# conv1D
loss:  4.605312824249268
accuracy :  0.009999999776482582
'''