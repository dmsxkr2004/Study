import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_iris()
print(datasets.feature_names)
# print(datasets)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
x = datasets.data
y = datasets.target
print(x)
print(y)
print(type(x))

# df = pd.DataFrame(x, columns = datasets.feature_names)
# df = pd.DataFrame(x, columns = datasets['feature_names'])

df = pd.DataFrame(x, columns=[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
print(df)
df['Target(Y)'] = y  # Y컬럼 추가

print(df)

print("--------------------------------상관계수 히트 맵----------------------------------")
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale = 1.2)
sns.heatmap(data = df.corr(), square=True, annot=True, cbar = True)

plt.show()