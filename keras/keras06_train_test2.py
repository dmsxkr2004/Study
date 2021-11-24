import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

### 과제 ###
# train과 test비율을 8:2으로 분리하시오.
x_train = x[:-2]
x_test = x[-2:]
y_train = y[:-2]
y_test = y[-2:]

print(x_train)
print(x_test)
print(y_train)
print(y_test)

'''
[1 2 3 4 5 6 7 8]
[ 9 10]
[1 2 3 4 5 6 7 8]
[ 9 10]
'''
#셋이나 솔트 값을 줘서 데이터들을 정렬해서 도출할수도 있을것 같음
#예측 값이기 때문에 무조건적인 정렬은 좋지않음
