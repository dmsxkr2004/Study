import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings('ignore')
#1. 데이터
path = "D:/Study/_data/"
datasets = pd.read_csv(path + "winequality-white.csv", sep=';', index_col=None, header=0)

x = datasets.drop(['fixed acidity','pH','quality'], axis = 1)
y = datasets['quality']
y = pd.get_dummies(y)
# for index, value in enumerate(y):
#     if value == 9:
#         y[index] = 7
#     elif value == 8:
#         y[index] = 7
#     elif value == 7:
#         y[index] = 7
#     elif value == 6:
#         y[index] = 6
#     elif value == 5:
#         y[index] = 5
#     elif value == 4:
#         y[index] = 5
#     elif value == 3:
#         y[index] = 5
#     else:
#         y[index] = 0
print(y.shape)
print(y)
############################################################################
# new_list = []

# for i in y:
#     if i <= 5:
#         new_list += [0]
#     elif i <= 7:
#         new_list += [2]
#     else:
#         new_list += [1]

# y = np.array(new_list)

# print(np.unique(y, return_counts = True))

############################################################################




print(np.unique(y, return_counts = True))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66, stratify=y)


print(y_train)
print(y_test)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,y_train.shape) # (3918, 9) (3918,)
print(x_test.shape, y_test.shape) # (980, 9) (980,)
#2. 모델구성
model = Sequential()
model.add(Dense(256, activation = 'relu', input_shape=(9,)))
model.add(Dense(128, activation = 'linear'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'linear'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))
#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.00001
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
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, filepath = model_path)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, mode = 'auto', verbose = 1, factor= 0.5)
start = time.time()
hist = model.fit(x_train, y_train, epochs=300, batch_size = 32, validation_split = 0.25, callbacks = [es, reduce_lr])
end = time.time()- start

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("걸린시간 : ", end)
print("learning_rate : ", learning_rate)
print('loss : ', loss[0])
print("acc : ", loss[1])

'''
걸린시간 :  120.96641087532043
learning_rate 1e-05
loss :  1.072222352027893
acc :  0.5469387769699097
'''