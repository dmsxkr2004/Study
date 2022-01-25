# 실습 : 다차원의 outlier가 출력되도록
import numpy as np
import matplotlib.pyplot as plt
aaa = np.array([[1, 2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
               [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]])
aaa = np.transpose(aaa)

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

outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

'''
1사분위 :  3.0
q2 :  6.0
3사분위 :  49.0
(array([ 2, 10], dtype=int64),)
'''

# 시각화
# 실습
# boxplot 으로 그려라 !!!
plt.boxplot(aaa)
plt.show()