import numpy as np
import pandas as pd 
import datetime
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '../_data/exam/samsung/'  
samsung = pd.read_csv(path + '삼성전자.csv',thousands=',',encoding = 'cp949')
kium = pd.read_csv(path + '키움증권.csv',thousands=',',encoding = 'cp949')

samsung = samsung.drop(range(25, 1120), axis=0)
kium = kium.drop(range(25, 1060), axis=0)

samsung = samsung.loc[::-1].reset_index(drop=True)
kium = kium.loc[::-1].reset_index(drop=True)

xx1 = samsung.drop(['일자', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis = 1) # axis=1 컬럼 삭제할 때 필요함
xx1 = np.array(xx1)
xx2 = kium.drop(['일자', '저가', '종가','전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis = 1)
xx2 = np.array(xx2)
#일자,시가,고가,저가,종가,전일비,Unnamed: 6,등락률,거래량,금액(백만),신용비,개인,기관,외인(수량),외국계,프로그램,외인비
print(xx1.shape,xx2.shape) #(22, 5) (22, 5)


def split_xy(dataset, time_steps, y_column):
    x, y =list(),list()
    for i in range(len(dataset)): 
        x_end_number = i + time_steps       
        y_end_number = x_end_number + y_column - 1                  
    
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :-1]
        tmp_y = dataset[x_end_number-1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
    
x1, y1 = split_xy(xx1,5,4)
x2, y2 = split_xy(xx2,5,4)

print(x1.shape,y1.shape,x2.shape,y2.shape) #(879, 15, 4) (879, 15) (879, 15, 5)

# print(x.columns, x.shape)  # (1060, 13)
  # (1060, 4) (1060,)

# x = x.to_numpy()

# x = x.head(10)
# y = y.head(20)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1,
        train_size =0.8, shuffle= True, random_state = 42)
x2_train, x2_test,y2_train,y2_test = train_test_split(x2,y2,
        train_size = 0.8, shuffle = True, random_state = 42)

print(x1_train.shape,y1_train.shape,x2_train.shape,y2_train.shape)  # (874, 20, 4) (874, 20)


# #2-1 모델1
# input1 = Input(shape=(5,1))   # (70,2) 
# dense1 = LSTM(64, activation='tanh')(input1)
# dense2 = Dense(64, activation='relu')(dense1)
# dense3 = Dense(52, activation='relu')(dense2)
# dense4 = Dense(40, activation='relu')(dense3)
# output1 = Dense(28, activation='relu')(dense4)


# #2-2 모델2
# input2 = Input(shape=(5,1))   # (70,3) 
# dense11 = LSTM(64, activation='tanh')(input2)
# dense12 = Dense(64, activation='relu')(dense11)
# dense13 = Dense(52, activation='relu')(dense12)
# dense14 = Dense(40, activation='relu')(dense13)
# output2 = Dense(28, activation='relu')(dense14)

# from tensorflow.keras.layers import Concatenate, concatenate  
# merge1 = Concatenate()([output1,output2]) 


# # 2-3 output모델1
# output21 = Dense(16)(merge1)
# output22 = Dense(10)(output21)
# output23 = Dense(5, activation='linear')(output22)
# last_output1 = Dense(4)(output23)  # y의 열의 갯수

# # 2-4 output모델2
# output31 = Dense(16)(merge1)
# output32 = Dense(10)(output31)
# output33 = Dense(5, activation='linear')(output32)
# last_output2 = Dense(4)(output33)




# # merge2 = Dense(10, activation='relu')(merge1)
# # merge3 = Dense(7)(merge2)
# # last_output = Dense(1)(merge3)

# model = Model(inputs = [input1,input2], outputs= ([last_output1,last_output2]))

# # model.summary()


# #3. 컴파일, 훈련
# model.compile(loss='mae', optimizer = 'adam')
# date = datetime.datetime.now()
# datetime = date.strftime("%m%d_%H%M") # month ,day , Hour, minite # 1206_0456
# filepath = './_ModelCheckPoint/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 2500 - 0.3724.hdf5
# model_path = "".join([filepath, 'samsung_kium_', datetime, '_', filename])
# es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
#                        filepath = model_path)
# start = time.time()
# model.fit([x1_train,x2_train], [y1_train,y2_train], epochs=500, batch_size=1,
#           validation_split=0.2, callbacks = [es, mcp])
# end = time.time()- start

# print("걸린시간 : ", round(end, 3), '초')
model = load_model('./_ModelCheckPoint/samsung_kium_1221_2037_0077-3926.7356.hdf5')


#4. 평가, 예측
loss = model.evaluate ([x1_test, x2_test], [y1_test,y2_test], batch_size=1)
print('loss :', loss) #loss :

result1 = np.array(xx1[-5:,1:].reshape(1,5,1))
result2 = np.array(xx2[-5:,1:].reshape(1,5,1))

y1_pred, y2_pred = model.predict([result1,result2])

print('삼성 시가 예측값 : ', y1_pred[-1][-1])
print('키움 시가 예측값 : ', y2_pred[-1][-1])


# model.save('../_data/exam/samsung/jechul_exam_{}.h5.format(y1_pred[-1])')
'''
Epoch 00241: val_loss did not improve from 3376.33984
Epoch 00241: early stopping
걸린시간 :  10.81 초
5/5 [==============================] - 0s 997us/step - loss: 2592.3423 - dense_11_loss: 1455.4531 - dense_15_loss: 1136.8890
loss : [2592.34228515625, 1455.453125, 1136.8890380859375]
삼성예측값 :  [76574.29]
키움예측값 :  [108301.32]
'''
'''
4/4 [==============================] - 1s 1ms/step - loss: 5606.3008 - dense_11_loss: 2868.7988 - dense_15_loss: 2737.5020
loss : [5606.30078125, 2868.798828125, 2737.501953125]
삼성 시가 예측값 :  78625.65
키움 시가 예측값 :  108635.06
'''
'''
4/4 [==============================] - 1s 0s/step - loss: 5114.2617 - dense_11_loss: 2321.8887 - dense_15_loss: 2792.3730
loss : [5114.26171875, 2321.888671875, 2792.373046875]
삼성 시가 예측값 :  78100.04
키움 시가 예측값 :  108336.52 
''' 
