
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape, Conv1D,LSTM
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.datasets import mnist


model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), strides = 1, # strides = 1 은 디폴트값 2를하면 2칸씩 건너뛰어서 자름
                 padding = 'same', input_shape = (28, 28, 1)))
model.add(MaxPooling2D())
#input_shape = (a, b, c)ㅡ> a - kernel_size + 1
# model.add(Conv2D(5, (3,3), activation = 'relu')) # 7,7,5
model.add(Conv2D(5, (2,2), activation = 'relu')) # 13, 13, 5
model.add(Dropout(0.2))
model.add(Conv2D(7, (2,2), activation = 'relu')) # 12, 12, 7
model.add(Conv2D(7, (2,2), activation = 'relu')) # 11, 11, 7
model.add(Conv2D(10, (2,2), activation = 'relu')) # 10, 10, 10
model.add(Flatten())                              # (N, 1000)
model.add(Reshape(target_shape = (100, 10))) # 타겟 쉐이프를 넣어주지않으면 오류가 뜬다. #(N, 100, 10)
model.add(Conv1D(5, 2))
model.add(LSTM(15))
model.add(Dense(10, activation = 'softmax'))
# model.add(Dense(64))

# model.add(Dense(16))
# model.add(Dense(5, activation = 'softmax'))
model.summary()
