# Градиентный бустинг

import numpy as np
import pandas
data_path = '.\\features.csv'
data = pandas.read_csv(data_path, index_col='match_id')

# Удаление признаков, связанных с итогами матча
X = data.drop(
    labels=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
            'barracks_status_radiant', 'barracks_status_dire']
    , axis=1)
# Вычленение целевой переменной
Y = data['radiant_win']

# Замена пропусков
X.fillna(value=0,inplace=True)

# Подготовка данных в виде матрицы и списка
X = np.matrix(X.as_matrix())
Y = np.matrix(Y.as_matrix())
Y = np.squeeze(np.asarray(Y))

# Кросс-валидатор
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

# Градиентный бустинг с использованием кросс-валидации и измерением времени
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import time

tree_nums = range(10,51,5)
precisions = []
elapsed_times = []

for tree_num in tree_nums:
    clf = GradientBoostingClassifier(n_estimators=tree_num, random_state=1)
    scores = []
    start_time = time.time()
    for train_index, test_index in k_fold.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(Y_test,Y_pred))
    elapsed_time = time.time() - start_time
    precision = sum(scores) / float(len(scores))
    print(tree_num, 'trees, precision is', precision, ', elapsed time is', elapsed_time, 'seconds')
    precisions.append(precision)
    elapsed_times.append(elapsed_time)

import matplotlib.pyplot as plt
# График точности на кросс-валидации
plt.plot(precisions)
plt.show()
# График времен работы на кросс-валидации
plt.plot(elapsed_times)
plt.show()
