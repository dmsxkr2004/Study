import numpy as np
# import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(x_train.shape)

# print(np.unique(y_train, return_counts=True))
scaler = MinMaxScaler()

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1
x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)
#2. 모델구성
model = Sequential()
model.add(Conv2D(1024, kernel_size=(2,2),activation = 'relu', input_shape = (32, 32, 3)))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(512, (2,2), activation = 'linear')) # 7,7,5
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(256, (2,2), activation = 'relu')) # 7,7,5
model.add(Flatten())
model.add(Dense(124, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'softmax'))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr = learning_rate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, filepath = model_path)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, mode = 'auto', verbose = 1, factor= 0.5)
start = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size = 32, validation_split = 0.25, callbacks = [es, reduce_lr, mcp])
end = time.time()- start

# model = load_model('./_ModelCheckPoint/mnist_1207_1843_0015-0.0745.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('learning_late : ', learning_rate)
print('accuracy : ', loss[1])
print("걸린시간 : ", round(end, 3), '초')

#평가지표 acc
#0.98 
# 발리데이션 스플릿으로 자르고
# 테스트로 
