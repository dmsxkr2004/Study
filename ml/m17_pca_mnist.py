import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist

(x_train, _ ), (x_test, _ ) = mnist.load_data()

print(x_train.shape, x_test.shape)

x = np.append(x_train, x_test, axis =0)
x = x.reshape(70000, 784)
print(x.shape) # (70000, 28, 28)


##################################################
# 실습
# pca를 통해 0.95 이상인 n_components 가 몇개??
##################################################

pca = PCA(n_components = 784)#n_components = 컬럼의 갯수를 의미한다.
x = pca.fit_transform(x)
print(x.shape)
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))
 
cumsum = np.cumsum(pca_EVR)
print(np.argmax(cumsum>= 0.95)+1)
print(np.argmax(cumsum>= 0.99)+1)
print(np.argmax(cumsum>= 0.999)+1)
print(np.argmax(cumsum) +1)
# print(cumsum)

# if cumsum >= 0.95:
#     print(cumsum)
#     print('누적합의 갯수 : ',len(cumsum))
    
import matplotlib.pyplot as plt
# plt.plot(cumsum)
# # plt.plot(pca_EVR)
# plt.grid()
# plt.show()
'''
154 # 0.95
331 # 0.99
486 # 0.999
713 # cumsum
'''