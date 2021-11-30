#none 이 있으면 결측치가 있음
import numpy as np
import pandas as pd # csv를 보게하는 api
path = "./_data/titanic/"

train = datasets = pd.read_csv(path + "train.csv", index_col = 0, header = 0)# csv = 엑셀파일과 같다
test = datasets = pd.read_csv(path + "test.csv", index_col = 0, header = 0)
gender_submission = pd.read_csv(path + "gender_submission.csv", index_col = 0, header = 0)

print(train.shape) # (891, 11)
print(test.shape) # (418, 10)
print(gender_submission.shape) # (418, 1)

#print(train.info())
print(train.describe())

# read_csv
# survived = y
