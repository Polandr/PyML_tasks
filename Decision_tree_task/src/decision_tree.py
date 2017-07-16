import pandas as P
data_path = '.\\titanic.csv'
data = P.read_csv(data_path, index_col='PassengerId',
                 usecols=['PassengerId','Pclass','Sex','Age','Fare','Survived'])
data.replace(to_replace=['male','female'],value=[0,1],inplace=True)
from sklearn import tree
import numpy as np
X = np.matrix(data.dropna().as_matrix(columns=['Pclass','Sex','Age','Fare']))
Y = np.matrix(data.dropna().as_matrix(columns=['Survived']))
clf = tree.DecisionTreeClassifier(random_state=241)
clf = clf.fit(X, Y)

importances = {}
for importance, feature_name in zip(clf.feature_importances_, ['Pclass','Sex','Age','Fare']):
    importances[feature_name] = importance
sorted_importance_names = sorted(importances, key=importances.get)
print('Two of the most important features are: ' + sorted_importance_names[-1] + ' and ' + sorted_importance_names[-2])
