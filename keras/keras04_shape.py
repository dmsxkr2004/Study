import numpy as np

a1 = np.array([[1, 2], [3, 4], [5, 6]])
a2 = np.array([[1, 2, 3], [4, 5, 6]])
a3 = np.array([[[1], [2], [3]], [[4], [5], [6]]])
a4 = np.array([[[1, 2], [3, 4]],[[5, 6], [7, 8]]])
a5 = np.array([[[1, 2, 3], [4, 5, 6]]])
a6 = np.array([1, 2, 3, 4, 5])
a7 = np.array([[[[1, 2, 3], [4, 5, 6]]]])
print(a1.shape)
print(a2.shape)
print(a3.shape)
print(a4.shape)
print(a5.shape)
print(a6.shape)
print(a7.shape)
print(type(a7))
'''
(3, 2)
(2, 3)
(2, 3, 1)
(2, 2, 2)
(1, 2, 3)
(5,)
(1, 1, 2, 3)
<class 'numpy.ndarray'>
넘파이 함수란 같은 데이터값을 지니고 있는 값을 배열화 하여 연산하기 편하게 하기 위한 함수값임
'''