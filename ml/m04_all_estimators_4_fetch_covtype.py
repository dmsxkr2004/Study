from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
import warnings
warnings.filterwarnings('ignore')

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
# allAlgorithms = all_estimators(type_filter = 'classifier')
allAlgorithms = all_estimators(type_filter = 'classifier')
print("allAlgorithms : ", allAlgorithms)
print("모델의 갯수 : ", len(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
    
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        # continue
        print(name, '은 에러 터진놈!!!')

'''
AdaBoostClassifier 의 정답률 :  0.5028613719095032
BaggingClassifier 의 정답률 :  0.9626515666549056
BernoulliNB 의 정답률 :  0.631833945767321
CalibratedClassifierCV 의 정답률 :  0.7122621619063191
CategoricalNB 의 정답률 :  0.6321437484402296
ComplementNB 의 정답률 :  0.6225742880992745
DecisionTreeClassifier 의 정답률 :  0.9396315069318348
DummyClassifier 의 정답률 :  0.48625250638968015
ExtraTreeClassifier 의 정답률 :  0.8680240613409292
ExtraTreesClassifier 의 정답률 :  0.9544331901930243
GaussianNB 의 정답률 :  0.09079800005163378
GradientBoostingClassifier 의 정답률 :  0.773499823584589
HistGradientBoostingClassifier 의 정답률 :  0.7815546930802131
KNeighborsClassifier 의 정답률 :  0.9376263951877318
LinearDiscriminantAnalysis 의 정답률 :  0.6797931206595355
LinearSVC 의 정답률 :  0.7124170632427734
LogisticRegression 의 정답률 :  0.7194220459024294
LogisticRegressionCV 의 정답률 :  0.7246370575630577
MLPClassifier 의 정답률 :  0.8393328915776701
MultinomialNB 의 정답률 :  0.6410247583969433
NearestCentroid 의 정답률 :  0.38585062347787924
PassiveAggressiveClassifier 의 정답률 :  0.5843308692546664
Perceptron 의 정답률 :  0.6010257910725196
QuadraticDiscriminantAnalysis 의 정답률 :  0.08459334096365842
RadiusNeighborsClassifier 의 정답률 :  0.6515064154970182
RandomForestClassifier 의 정답률 :  0.9556465839952497
RidgeClassifier 의 정답률 :  0.7008510967875184
RidgeClassifierCV 의 정답률 :  0.7008424911577154
SGDClassifier 의 정답률 :  0.7109282892868515
SVC 의 정답률 :  0.773955921964149
'''