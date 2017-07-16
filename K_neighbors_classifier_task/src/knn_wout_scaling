import pandas as P
data_path = '.\\wine.data'
features = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
           'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
           'Color intensity', 'Hue', 'OD280/OD315 of diluted wines','Proline']
target = ['Grape type']
data = P.read_csv(data_path, header=None, names=target+features)

import numpy as np
X = np.matrix(data.dropna().as_matrix(columns=features))
Y = np.matrix(data.dropna().as_matrix(columns=target))
Y = np.squeeze(np.asarray(Y))

from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

precisions = []
for k in range(1,50):
    k_neghbors = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(k_neghbors, X, Y, cv=k_fold)
    precisions.append(scores.mean())

import matplotlib.pyplot as plt
plt.plot(precisions)
plt.axis([0,49,0.0,1.0])
plt.show()
print('Best precision without scaling is %.2f at value %d' % (max(precisions), precisions.index(max(precisions))+1))
