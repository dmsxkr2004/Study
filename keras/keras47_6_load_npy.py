import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler # 전처리 4대장
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# np.save('./_save_npy/keras47_5_train_x.npy', arr=xy_train[0][0])
# np.save('./_save_npy/keras47_5_train_y.npy', arr=xy_train[0][1])
# np.save('./_save_npy/keras47_5_test_x.npy', arr=xy_test[0][0])
# np.save('./_save_npy/keras47_5_test_y.npy', arr=xy_test[0][1])

x_train = np.load('./_save_npy/keras47_5_train_x.npy')
x_test = np.load('./_save_npy/keras47_5_test_x.npy')
y_train = np.load('./_save_npy/keras47_5_train_y.npy')
y_test = np.load('./_save_npy/keras47_5_test_y.npy')
# print(x_train)
print(x_train.shape)# (160, 150, 150, 3)
print(x_test.shape)# (160, 150, 150, 3)
print(y_train.shape) # (160,)
print(y_test.shape) # (160,)

#2. 모델구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(150, 150, 3)))
model.summary()
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )               # 이진분류 인지 - binary_crossentropy

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto', verbose=1, restore_best_weights=True)    

model.fit(x_train, y_train, epochs=100, batch_size=5, verbose=1, validation_split=0.2, callbacks=[es])



# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])