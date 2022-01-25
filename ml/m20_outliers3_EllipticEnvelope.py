import numpy as np

aaa = np.array([[1, 2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 1000, 1001, 1002, 99]])

# (2, 13) - > (13, 2)
aaa = np.transpose(aaa)  # (13, 2)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1)

outliers.fit(aaa)

pred = outliers.predict(aaa)
print(pred) # [ 1  1  1  1  1  1  1  1  1 -1 -1  1  1]  -1로 표기되어 있는 인덱스가 이상치로 판단

# 명재형님 소스
b = list(pred)
print(b.count(-1))
index_for_outlier = np.where(pred == -1)
print('outier indexes are', index_for_outlier)
outlier_value = aaa[index_for_outlier]
print('outlier_value :', outlier_value)