from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time
# import matplotlib.pyplot as plt

#1.데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape)# (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)# (10000, 32, 32, 3) (10000, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(50000, 10) (10000, 10)

scaler = StandardScaler()

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1
# x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1))#.reshape(x_test.shape)

#2. 모델구성
model = Sequential()
# model.add(Conv2D(6, kernel_size=(2,2), strides = 1,
#                  padding='same', input_shape = (32, 32, 3))) #padding = same, valid
# model.add(Conv2D(4, (2,2), activation = 'relu')) # 7,7,5
# model.add(MaxPooling2D())
# model.add(Dropout(0.2))
# model.add(Conv2D(2, (2,2), activation = 'relu')) # 7,7,5
# model.add(Flatten())
model.add(Dense(64, activation = 'relu', input_shape = (3072, )))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
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
model_path = "".join([filepath, 'cifar10_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=5, mode = 'auto', verbose=1, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=32, batch_size = 32, validation_split = 0.1, callbacks = [es,mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/cifar10_1208_2340_0022-1.2690.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy : ', loss[1])
# plt.imshow(x_train[0])
# plt.show()
'''
loss:  1.3080413341522217
accuracy :  0.5320000052452087
'''