import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

#import data into pandas dataframe
dataset = pd.read_csv("D:/Study/_data/winequality-white.csv", sep = ';')

#display first 5 lines

# print(dataset)
#print data properties
dataset.groupby( [ "quality"] ).count().plot(kind='bar', rot=0)
plt.show()