import pandas as pd
import os
import os.path as osp
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

data_dir = 'D:/_data/dacon/housing/'

train = pd.read_csv(osp.join(data_dir, 'train.csv'))
test = pd.read_csv(osp.join(data_dir, 'test.csv'))

train.drop('id', axis=1, inplace=True) # id 제거
test.drop('id', axis=1, inplace=True) # id 제거
print(train.shape, test.shape)

train.head()

# 중복값 제거
print("제거 전 :", train.shape)
train = train.drop_duplicates()
print("제거 후 :", train.shape)

# train[train['Garage Yr Blt']> 2050] # 254
train.loc[254, 'Garage Yr Blt'] = 2007
train.loc[[219,413,696,934,1172], 'Full Bath'] = 1
# test.loc[[113,422,595,641,1309], 'Full Bath'] = 1
# 품질 관련 변수 → 숫자로 매핑
qual_cols = train.dtypes[train.dtypes == np.object].index
def label_encoder(df_, qual_cols):
  df = df_.copy()
  mapping={
      'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':2
  }
  for col in qual_cols :
    df[col] = df[col].map(mapping)
  return df

train = label_encoder(train, qual_cols)
test = label_encoder(test, qual_cols)
train.head()

def feature_eng(data_):
    data = data_.copy()
    data['Year Gap Remod'] = data['Year Remod/Add'] - data['Year Built']
    data['Car Area'] = data['Garage Area']/data['Garage Cars']
    data['2nd flr SF'] = data['Gr Liv Area'] - data['1st Flr SF']
    data['2nd flr'] = data['2nd flr SF'].apply(lambda x : 1 if x > 0 else 0)
    data['Total SF'] = data[['Gr Liv Area',"Garage Area", "Total Bsmt SF"]].sum(axis=1)
    data['Sum Qual'] = data[["Exter Qual", "Kitchen Qual", "Overall Qual"]].sum(axis=1)
    data['Garage InOut'] = data.apply(lambda x : 1 if x['Gr Liv Area'] != x['1st Flr SF'] else 0, axis=1)
    return data

train = feature_eng(train)
test = feature_eng(test)

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor, Pool
from ngboost import NGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

# 평가 기준 정의
def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

nmae_score = make_scorer(NMAE, greater_is_better=False)
kf = KFold(n_splits = 10, random_state = 42, shuffle = True)

X = train.drop(['target'], axis = 1)
y = np.log1p(train.target)

target = test[X.columns]

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

# LinearRegression
lr_pred = np.zeros(target.shape[0])
lr_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    lr = LinearRegression(normalize=True)
    lr.fit(tr_x, tr_y)
    
    val_pred = np.expm1(lr.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    lr_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = lr.predict(target) / 10
    lr_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(lr_val)} & std = {np.std(lr_val)}')

# Ridge
rg_pred = np.zeros(target.shape[0])
rg_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    rg = Ridge()
    rg.fit(tr_x, tr_y)
    
    val_pred = np.expm1(rg.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    rg_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = rg.predict(target) / 10
    rg_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(rg_val)} & std = {np.std(rg_val)}')

# Lasso
ls_pred = np.zeros(target.shape[0])
ls_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    ls = Lasso()
    ls.fit(tr_x, tr_y)
    
    val_pred = np.expm1(ls.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    ls_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = ls.predict(target) / 10
    ls_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(ls_val)} & std = {np.std(ls_val)}')

# ElasticNet
el_pred = np.zeros(target.shape[0])
el_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    el = ElasticNet()
    el.fit(tr_x, tr_y)
    
    val_pred = np.expm1(el.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    el_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = el.predict(target) / 10
    el_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(el_val)} & std = {np.std(el_val)}')

# GradientBoostingRegressor
gbr_pred = np.zeros(target.shape[0])
gbr_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    gbr = GradientBoostingRegressor(random_state = 42, max_depth = 4, learning_rate = 0.05, n_estimators = 1000)
    gbr.fit(tr_x, tr_y)
    
    val_pred = np.expm1(gbr.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    gbr_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = gbr.predict(target) / 10
    gbr_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(gbr_val)} & std = {np.std(gbr_val)}')

# RandomForestRegressor
rf_pred = np.zeros(target.shape[0])
rf_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    rf = RandomForestRegressor(random_state = 42, criterion = 'mae')
    rf.fit(tr_x, tr_y)
    
    val_pred = np.expm1(rf.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    rf_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    fold_pred = rf.predict(target) / 10
    rf_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(rf_val)} & std = {np.std(rf_val)}')

# NGBRegressor
ngb_pred = np.zeros(target.shape[0])
ngb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    ngb = NGBRegressor(random_state = 42, n_estimators = 1000, verbose = 0, learning_rate = 0.03)
    ngb.fit(tr_x, tr_y, val_x, val_y, early_stopping_rounds = 300)
    
    val_pred = np.expm1(ngb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    ngb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = ngb.predict(target) / 10
    ngb_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(ngb_val)} & std = {np.std(ngb_val)}')

# Catboost
cb_pred = np.zeros(target.shape[0])
cb_val = []
for n, (tr_idx, val_idx) in enumerate(kf.split(X, y)) :
    print(f'{n + 1} FOLD Training.....')
    tr_x, tr_y = X.iloc[tr_idx], y.iloc[tr_idx]
    val_x, val_y = X.iloc[val_idx], np.expm1(y.iloc[val_idx])
    
    tr_data = Pool(data = tr_x, label = tr_y)
    val_data = Pool(data = val_x, label = val_y)
    
    cb = CatBoostRegressor(depth = 4, random_state = 42, loss_function = 'MAE', n_estimators = 3000, learning_rate = 0.03, verbose = 0)
    cb.fit(tr_data, eval_set = val_data, early_stopping_rounds = 750, verbose = 1000)
    
    val_pred = np.expm1(cb.predict(val_x))
    val_nmae = NMAE(val_y, val_pred)
    cb_val.append(val_nmae)
    print(f'{n + 1} FOLD NMAE = {val_nmae}\n')
    
    target_data = Pool(data = target, label = None)
    fold_pred = cb.predict(target) / 10
    cb_pred += fold_pred
print(f'10FOLD Mean of NMAE = {np.mean(cb_val)} & std = {np.std(cb_val)}')

# 검증 성능 확인하기
val_list = [lr_val, rg_val, ls_val, el_val, gbr_val, rf_val, ngb_val, cb_val]
for val in val_list :
  print("{:.8f}".format(np.mean(val)))

# submission 파일에 입력
sub = pd.read_csv(osp.join(data_dir, 'sample_submission.csv'))
sub['target'] = np.expm1((ngb_pred + rf_pred + rg_pred + gbr_pred) / 4)
sub['target']

# csv 파일로 내보내기
sub_dir = 'D:/_data/dacon/housing/'
sub.to_csv(osp.join(sub_dir, 'jechul.csv'), index=False)