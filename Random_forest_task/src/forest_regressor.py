import pandas as P
data_path = '.\\abalone.csv'
data = P.read_csv(data_path)
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

import numpy as np
X = np.matrix(data.dropna().as_matrix(columns=['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight',
                                               'ShuckedWeight', 'VisceraWeight', 'ShellWeight']))
Y = np.matrix(data.dropna().as_matrix(columns=['Rings']))
Y = np.squeeze(np.asarray(Y))

from sklearn.model_selection import KFold
k_fold = KFold(n_splits=5, shuffle=True, random_state=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer

precisions = []
for tree_num in range(1,51):
    forest = RandomForestRegressor(n_estimators=tree_num, random_state=1)
    scores = cross_val_score(forest, X, Y, scoring=make_scorer(r2_score), cv=k_fold)
    precision = scores.mean()
    precisions.append(precision)

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(precisions)
plt.axis([0.,50.,0.,1.])

precision_threshold = 0.52
idx = 1
for precision in precisions:
    if precision > precision_threshold:
        break
    idx += 1
print('Minimal precision more than 0.52 is %f at %d trees' % (precision, idx))
