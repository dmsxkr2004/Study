from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, LabelEncoder, OneHotEncoder
from pandas import get_dummies
from sklearn import metrics
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.saving.save import load_model

#1. 데이터
path = "../_data/dacon/wine/"
train = pd.read_csv(path +"train.csv")
test_file = pd.read_csv(path + "test.csv") 
submission = pd.read_csv(path+"sample_Submission.csv")

x = train.drop(['id', 'quality'], axis =1)
y = train['quality']

le = LabelEncoder()
label = x['type']
le.fit(label)
x['type'] = le.transform(label)
test_file = test_file.drop(['id'], axis=1)
label = test_file['type']
le.fit(label)
test_file['type'] = le.transform(label)
y = get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.9, shuffle = True, random_state = 39)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_file = scaler.transform(test_file)

#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim = 12))# linear= 기본값
model.add(Dense(35))
model.add(Dropout(0.5))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(8))
model.add(Dense(5, activation = 'softmax'))

#3. 컴파일, 훈련
'''
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # categorical_crossentorpy 결과값이[0~3]이상일때 적합한 값 
                                                                                        # metrics 몇개가 맞았는지 결과값을 보기위해 씀 
                                                                                        # # 리스트[]는 다른 값이 더 들어갈수도 있음
es = EarlyStopping(monitor='val_loss', patience=100, mode = 'auto', restore_best_weights=True)
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size = 10, 
                 validation_split = 0.1 , callbacks = [es], verbose = 1)
end = time.time()- start

print("걸린시간 : ", round(end, 3), '초')
'''
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
model_path = "".join([filepath, 'wine_', datetime, '_', filename])
                # ./_ModelCheckPoint/1206_0456_2500-0.3724.hdf5
############################################################################

es = EarlyStopping(monitor= 'val_loss', patience=150, mode = 'auto', verbose=1)#, restore_best_weights = True)
mcp = ModelCheckpoint(monitor='val_loss',mode='auto', verbose = 1, save_best_only=True, 
                      filepath = model_path)
start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size = 5, validation_split = 0.1, callbacks = [es,mcp])
end = time.time()- start
print("걸린시간 : ", round(end, 3), '초')
# model = load_model('./_ModelCheckPoint/wine_1206_2248_0060-0.9007.hdf5')
#4 평가예측
loss = model.evaluate(x_test,y_test)
print("loss : ",loss[0])
print("accuracy : ", loss[1])

################################ 제출용 ########################################
result = model.predict(test_file)
print(result[:5])
result_recover = np.argmax(result, axis=1).reshape(-1,1) + 4
print(result_recover[:5])
print(np.unique(result_recover))    # value_counts = pandas에서만 먹힌다.
submission['quality'] = result_recover

# print(submission[:10])
submission.to_csv(path + "jechul6.csv", index = False)
print(result_recover)
'''
걸린시간 :  169.211 초
21/21 [==============================] - 0s 1ms/step - loss: 1.0165 - accuracy: 0.5750
loss :  1.0164984464645386
accuracy :  0.5749613642692566
'''
'''
loss :  0.981419026851654
accuracy :  0.5709876418113708
'''
'''
loss :  0.9854784607887268
accuracy :  0.5895061492919922
'''