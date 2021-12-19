import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

path = '../_data/exam/samsung/'

samsung = pd.read_csv(path + '삼성전자.csv', encoding = 'cp949',thousands=',')
kium = pd.pred_csv(path + '키움증권.csv', encoding = 'cp949', thousands = ',')
print(samsung.columns) # (1120, 17)

x1 = samsung.drop(['일자', '전일비','종가', 'Unnamed: 6', '등락률','신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],axis = 1)
x2 = kium.drop(['일자', '전일비','종가', 'Unnamed: 6', '등락률','신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],axis = 1)
# y1 = samsung['종가']
x1_train, x1_test,x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, train_size=0.8, random_state=42)

print(type(x1_train))

# print(x1_train.shape, x1_test.shape)# (896, 17) (224, 17)
# print(x2_train.shape, x2_test.shape)# (848, 17) (212, 17)
