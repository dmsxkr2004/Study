from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time
#실습!!!
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = StandardScaler()

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1
# x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1
# print(x_train.shape)
# print(y_train.shape)
# print(y_test.shape)

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1))#.reshape(x_test.shape)

#2. 모델 구성
model = Sequential()
# model.add(Dense(64, input_shape = (28*28, )))
model.add(Dense(64, input_shape = (784, )))
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


