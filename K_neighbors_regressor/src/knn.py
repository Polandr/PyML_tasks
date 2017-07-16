from sklearn import datasets
boston = datasets.load_boston()
X = boston.data
Y = boston.target
from sklearn import preprocessing
preprocessing.scale(X,copy=False)

from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from numpy import linspace

precisions = []
for p in linspace(1,10, num=200):
    k_neghbors = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    scores = cross_val_score(k_neghbors, X, Y, scoring='neg_mean_squared_error', cv=k_fold)
    precisions.append(scores.mean())

import matplotlib.pyplot as plt
dst = max(precisions) - min(precisions)
upper_bound, lower_bound = max(precisions) + 0.2*dst, min(precisions) - 0.2*dst
plt.plot(precisions)
plt.axis([0,len(precisions),lower_bound,upper_bound])
plt.show()
print('Best precision is %.2f at value %d' % (max(precisions), precisions.index(max(precisions))+1))
