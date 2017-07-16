# Логистическая регрессия без категориальных признаков

import numpy as np
import pandas
data_path = '.\\features.csv'
data = pandas.read_csv(data_path, index_col='match_id')

# Удаление признаков, связанных с итогами матча
X = data.drop(
    labels=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
            'barracks_status_radiant', 'barracks_status_dire']
    , axis=1)

# Удаление категориальных признаков
import re
excl_features = ['lobby_type'] + \
[name for name in data.columns if re.search('r._hero', name)] + \
[name for name in data.columns if re.search('d._hero', name)]
X = X.drop(labels=excl_features, axis=1)

# Вычленение целевой переменной
Y = data['radiant_win']

# Замена пропусков
X.fillna(value=0,inplace=True)

# Подготовка данных в виде матрицы и списка
X = np.matrix(X.as_matrix())
Y = np.matrix(Y.as_matrix())
Y = np.squeeze(np.asarray(Y))

# Масштабирование
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=False)

# Кросс-валидатор
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

# Логистическая регрессия с использованием кросс-валидации и измерением времени
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import time

penalty_weights = np.arange(0.001,0.015,0.001)
precisions = []
elapsed_times = []

for penalty_weight in penalty_weights:
    reg = LogisticRegression(penalty='l2', C=penalty_weight, random_state=1)
    scores = []
    start_time = time.time()
    for train_index, test_index in k_fold.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        scaler.fit_transform(X_train)
        reg.fit(X_train, Y_train)
        scaler.transform(X_test)
        Y_pred = reg.predict_proba(X_test)[:, 1]
        scores.append(roc_auc_score(Y_test,Y_pred))
    elapsed_time = time.time() - start_time
    precision = sum(scores) / float(len(scores))
    print('Penalty weight is ', penalty_weight, ', precision is', precision, ', elapsed time is', elapsed_time, 'seconds')
    precisions.append(precision)
    elapsed_times.append(elapsed_time)

import matplotlib.pyplot as plt
# График точности на кросс-валидации
plt.plot(precisions)
plt.show()
# График времен работы на кросс-валидации
plt.plot(elapsed_times)
plt.show()
# Максимальное значение точности и соответствующее значение коэффициента С
print('Maximum precision is', max(precisions), ', best C is', penalty_weights[precisions.index(max(precisions))])
