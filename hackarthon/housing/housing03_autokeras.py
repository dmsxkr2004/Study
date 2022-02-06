from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
import autokeras as ak
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from xgboost import XGBRegressor
def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score    
############### 이상치 확인 처리 ###################
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) |
                    (data_out<lower_bound))
# outliers_loc = outliers(aaa)
# print("이상치의 위치 : ", outliers_loc)
############### 이상치 확인 처리 ###################
path = "D:/_data/dacon/housing/"
datasets = pd.read_csv(path + "train.csv", index_col = 0, header = 0)
test_sets = pd.read_csv(path + "test.csv", index_col = 0, header = 0)
sub_sets = pd.read_csv(path + "sample_submission.csv", index_col = 0, header = 0)

print(datasets.info()) # info 에서 object 는 문자열로 생각하면 편하다. 예외사항도 있음
'''
 #   Column          Non-Null Count  Dtype
---  ------          --------------  -----
 0   Overall Qual    1350 non-null   int64
 1   Gr Liv Area     1350 non-null   int64
 2   Exter Qual      1350 non-null   object
 3   Garage Cars     1350 non-null   int64
 4   Garage Area     1350 non-null   int64
 5   Kitchen Qual    1350 non-null   object
 6   Total Bsmt SF   1350 non-null   int64
 7   1st Flr SF      1350 non-null   int64
 8   Bsmt Qual       1350 non-null   object
 9   Full Bath       1350 non-null   int64
 10  Year Built      1350 non-null   int64
 11  Year Remod/Add  1350 non-null   int64
 12  Garage Yr Blt   1350 non-null   int64
 13  target          1350 non-null   int64
'''
print(datasets.describe())
'''
       Overall Qual  Gr Liv Area  Garage Cars  Garage Area  Total Bsmt SF   1st Flr SF    Full Bath   Year Built  Year Remod/Add  Garage Yr Blt         target
count   1350.000000  1350.000000  1350.000000  1350.000000    1350.000000  1350.000000  1350.000000  1350.000000     1350.000000    1350.000000    1350.000000
mean       6.208889  1513.542222     1.870370   502.014815    1082.644444  1167.474074     1.560741  1972.987407     1985.099259    1978.471852  186406.312593
std        1.338015   487.523239     0.652483   191.389956     384.067713   375.061407     0.551646    29.307257       20.153244      25.377278   78435.424758
min        2.000000   480.000000     1.000000   100.000000     105.000000   480.000000     0.000000  1880.000000     1950.000000    1900.000000   12789.000000
25%        5.000000  1144.000000     1.000000   368.000000     816.000000   886.250000     1.000000  1955.000000     1968.000000    1961.000000  135000.000000
50%        6.000000  1445.500000     2.000000   484.000000    1009.000000  1092.500000     2.000000  1976.000000     1993.000000    1978.500000  165375.000000
75%        7.000000  1774.500000     2.000000   588.000000    1309.500000  1396.500000     2.000000  2002.000000     2004.000000    2002.000000  217875.000000
max       10.000000  4476.000000     5.000000  1488.000000    2660.000000  2898.000000     4.000000  2010.000000     2010.000000    2207.000000  745000.000000
'''
print(datasets.isnull().sum()) # null 없음

############### 중복값 처리########################
print("중복값 제거 전 : ", datasets.shape)
datasets = datasets.drop_duplicates()
print("중복값 제거 후 : ", datasets.shape)
############### 중복값 처리########################
############### 이상치 확인 처리 ###################
outliers_loc = outliers(datasets['Garage Yr Blt'])
print(outliers_loc)
print(datasets.loc[[255], 'Garage Yr Blt']) # 2207
datasets.drop(datasets[datasets['Garage Yr Blt']==2207].index, inplace = True) # 행 하나 드랍

############### 이상치 확인 처리 ###################
print(datasets['Exter Qual'].value_counts())

'''
Overall Qual      0
Gr Liv Area       0
Exter Qual        0
Garage Cars       0
Garage Area       0
Kitchen Qual      0
Total Bsmt SF     0
1st Flr SF        0
Bsmt Qual         0
Full Bath         0
Year Built        0
Year Remod/Add    0
Garage Yr Blt     0
target            0
dtype: int64
'''
print(datasets['Bsmt Qual'].value_counts()) # datasets
# TA    808
# Gd    485
# Ex     49
# Fa      8
# Name: Exter Qual, dtype: int64
# TA    660
# Gd    560
# Ex    107
# Fa     23
# Name: Kitchen Qual, dtype: int64
# TA    605
# Gd    582
# Ex    134
# Fa     28
# Po      1
# Name: Bsmt Qual, dtype: int64
print(test_sets['Bsmt Qual'].value_counts()) # test_sets
# TA    794
# Gd    489
# Ex     58
# Fa      9
# Name: Exter Qual, dtype: int64
# TA    666
# Gd    566
# Ex     94
# Fa     23
# Po      1
# Name: Kitchen Qual, dtype: int64
# Gd    597
# TA    582
# Ex    124
# Fa     46
# Po      1
# Name: Bsmt Qual, dtype: int64
qual_cols = datasets.dtypes[datasets.dtypes == np.object].index
def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={
      'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2 # test_sets 데이터에 Po값이 2개가 있어서 Po 를 1 ㅡ> 2 처리해줌
  }
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

datasets = label_encoder(datasets, qual_cols)
test_sets = label_encoder(test_sets, qual_cols)
print(datasets.head())
print(datasets.shape) # (1350, 14)
print(test_sets.shape) # (1350, 13)

'''
    Overall Qual  Gr Liv Area  Exter Qual  Garage Cars  Garage Area  Kitchen Qual  Total Bsmt SF  1st Flr SF  Bsmt Qual  Full Bath  Year Built  Year Remod/Add  Garage Yr Blt  target    
id
1             10         2392           5            3          968             5           2392        2392          5          2        2003            2003           2003  386250    
2              7         1352           4            2          466             4           1352        1352          5          2        2006            2007           2006  194000    
3              5          900           3            1          288             3            864         900          3          1        1967            1967           1967  123000    
4              5         1174           3            2          576             4            680         680          3          1        1900            2006           2000  135000    
5              7         1958           4            3          936             4           1026        1026          4          2        2005            2005           2005  250000    
'''
#######################################################분류형 컬럼을 one hot encoding##########################################################
datasets = pd.get_dummies(datasets, columns=['Exter Qual','Kitchen Qual','Bsmt Qual'])

test_sets = pd.get_dummies(test_sets, columns=['Exter Qual','Kitchen Qual','Bsmt Qual'])
#######################################################분류형 컬럼을 one hot encoding##########################################################
# print(datasets.columns)
# print(test_sets.columns)

print(datasets.shape) # (1350, 23)
print(test_sets.shape) # (1350, 22)

x = datasets.drop(['target'], axis = 1)
y = datasets['target']

test_sets = test_sets.to_numpy()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, train_size = 0.8, shuffle = True)
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)
print(x_train.shape, y_train.shape) # (1080, 22) (1080,)
print(x_test.shape, y_test.shape) # (270, 22) (270,)

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer(method = 'box-cox') # error
scaler = PowerTransformer(method = 'yeo-johnson') # 디폴트

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_sets = scaler.transform(test_sets)

ak_model = ak.StructuredDataRegressor(
    overwrite=True, max_trials=10, loss = 'mean_absolute_error'
)
start = time.time()
ak_model.fit(x_train,y_train)#, epochs=200, validation_split=0.2)
end = time.time() - start
model = ak_model.export_model() # trial의 수만큼 훈련 시킨 것 중에 가장 좋은 것을 꺼낸다.
# save model
#4. 평가, 예측
y_predict = ak_model.predict(x_test)

results = model.evaluate(x_test, y_test)
print("loss : ", np.round(results, 4))

print(y_test.shape, y_predict.shape) # (270,) (270, 1)
y_predict = y_predict.reshape(270,)

nmae = NMAE(np.expm1(y_test), np.expm1(y_predict))
print("NMAE : ", np.round(nmae, 4))


####################### 제출용 #############################
colsample_bytree= 0.6460
learning_rate= 0.1078
max_depth= 5
min_child_weight= 0.2615
n_estimators= 9145
reg_lamda= 2.9871
subsample= 0.5156

y_submit = model.predict(test_sets)

y_submit = np.expm1(y_submit)
sub_sets.target = y_submit

path_save_csv = "D:/Study/hackarthon/_save_csv/"
now1 = datetime.now()
now_date = now1.strftime("%m%d_%H%M")

sub_sets.to_csv(path_save_csv + now_date + '_' + str(round(nmae, 4))+'.csv')

model.summary()

with open(path_save_csv + now_date + '_' + 
          str(round(nmae, 4)) + 'submit.txt','a') as file :
        file.write("\n=========================")
        file.write('저장시간 : '+ now_date + '\n')
        file.write('scaler : ' + str(scaler) + '\n')
        file.write('colsample_bytree : ' + str('colsample_bytree') + '\n')
        file.write('learning_rate : ' + str('learning_rate') + '\n')
        file.write('max_depth : ' + str('max_depth') + '\n')
        file.write('min_child_weight : ' + str('min_child_weight') + '\n')
        file.write('n_estimators : ' + str('n_estimators') + '\n')
        file.write('reg_lamda : ' + str('reg_lamda') + '\n')
        file.write('subsample : ' + str('subsample') + '\n')
        
        file.write('걸린시간 : ' + str(round(end, 4)) + '\n')
        file.write('evaluate : ' +str(np.round(results, 4)) + '\n')
        file.write('NMAE : ' + str(round(nmae, 4)) + '\n')
        
        #file.close() # 파일을 열면 닫아주어야함 with 써서 안닫아주어도 됨