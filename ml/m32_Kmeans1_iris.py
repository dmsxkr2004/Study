from sklearn.datasets import load_iris
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
datasets = load_iris()

# x = datasets.data
# y = datasets.target

irisDF = pd.DataFrame(datasets.data)#datasets.feature_names
# y = datasets.target
# print(type(x)) # <class 'pandas.core.frame.DataFrame'>
print(irisDF)

kmeans = KMeans(n_clusters=3, random_state=66)
kmeans.fit(irisDF)

print(np.sort(kmeans.labels_))
# print(y)
y = np.sort(kmeans.labels_)

irisDF['cluster'] = kmeans.labels_
irisDF['target'] = datasets.target
# irisDF = irisDF.drop(['cluster','target'], axis = 1)
# print(irisDF)


# results = kmeans.score(irisDF, irisDF['target'])
# print("results : ", results)

# y_pred = kmeans.predict(irisDF['cluster'])
acc = accuracy_score(irisDF['target'],irisDF['cluster'])
print('acc : ', acc)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''