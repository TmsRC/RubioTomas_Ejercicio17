import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
from scipy.io import arff


data = arff.loadarff('1year.arff')
df = pd.DataFrame(data[0])

data = arff.loadarff('2year.arff')
df2 = pd.DataFrame(data[0])
data = arff.loadarff('3year.arff')
df3 = pd.DataFrame(data[0])
data = arff.loadarff('4year.arff')
df4 = pd.DataFrame(data[0])
data = arff.loadarff('5year.arff')
df5 = pd.DataFrame(data[0])

df = df.append(df2,ignore_index=True)
df = df.append(df3,ignore_index=True)
df = df.append(df4,ignore_index=True)
df = df.append(df5,ignore_index=True)

target = 1*np.array(df['class']==b'1')

df = df.drop(['class'],axis=1)
predictors = list(df.keys())
X = np.array(df,dtype=float)

target = target[~np.isnan(X).any(axis=1)]
X = X[~np.isnan(X).any(axis=1)]

X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(X,target,test_size=0.5)
X_test,X_validation,y_test,y_validation = sklearn.model_selection.train_test_split(X_test,y_test,test_size=0.4)


n_trees = np.arange(1,15,1) #Ya vi que el máximo debe estar entre 1 y 15
f1_test = []

for i, n_tree in enumerate(n_trees):
    #print(n_tree)
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(X_train, y_train)
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test)))
    

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_trees[np.argmax(f1_test)], max_features='sqrt')
clf.fit(X_train, y_train)
pesos = clf.feature_importances_
f1 = sklearn.metrics.f1_score(y_validation, clf.predict(X_validation))

plt.figure()
a = pd.Series(pesos, index=predictors)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')
plt.title('M = {}, f1_score = {:.2f}'.format(n_trees[np.argmax(f1_test)],f1))
plt.savefig('features.png')