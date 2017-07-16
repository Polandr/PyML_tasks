# Лучший из двух методов - логистическая регрессия с параметром регуляризации равным 0.005

import numpy as np
import pandas
train_data_path = '.\\features.csv'
test_data_path = '.\\features_test.csv'
train_data = pandas.read_csv(train_data_path, index_col='match_id')
test_data = pandas.read_csv(test_data_path, index_col='match_id')

# Удаление признаков, связанных с итогами матча для тренировочных данных и копирование тестовых данных
X_train = train_data.drop(
    labels=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
            'barracks_status_radiant', 'barracks_status_dire']
    , axis=1)
X_test = test_data

# Подсчет количества различных героев
import re
players = [name for name in train_data.columns if re.search('r._hero', name)] + \
[name for name in train_data.columns if re.search('d._hero', name)]
train_heroes = train_data[players].stack().value_counts()
test_heroes = test_data[players].stack().value_counts()
heroes_num = max(max(train_heroes.index), max(test_heroes.index))

# Создание "мешка слов"
X_train_pick = np.zeros((train_data.shape[0], heroes_num))
for i, match_id in enumerate(train_data.index):
    for p in range(5):
        X_train_pick[i, train_data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_train_pick[i, train_data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_test_pick = np.zeros((test_data.shape[0], heroes_num))
for i, match_id in enumerate(test_data.index):
    for p in range(5):
        X_test_pick[i, test_data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_test_pick[i, test_data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
        
# Удаление категориальных признаков
excl_features = ['lobby_type'] + players
X_train = X_train.drop(labels=excl_features, axis=1)
X_test = X_test.drop(labels=excl_features, axis=1)

# Вычленение целевой переменной
Y_train = train_data['radiant_win']

# Замена пропусков
X_train.fillna(value=0,inplace=True)
X_test.fillna(value=0,inplace=True)

# Подготовка данных в виде матриц и списка
X_train = np.matrix(X_train.as_matrix())
X_test = np.matrix(X_test.as_matrix())
Y_train = np.matrix(Y_train.as_matrix())
Y_train = np.squeeze(np.asarray(Y_train))

# Добавление "мешка слов" по героям
X_train = np.hstack((X_train,X_train_pick))
X_test = np.hstack((X_test,X_test_pick))

# Масштабирование
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=False)
scaler.fit_transform(X_train)
scaler.transform(X_test)

# Логистическая регрессия над тестовыми данными
from sklearn.linear_model import LogisticRegression

reg = LogisticRegression(penalty='l2', C=0.005, random_state=1)
reg.fit(X_train, Y_train)
Y_pred = reg.predict_proba(X_test)[:, 1]

print('Maximum probability is', max(Y_pred), 'minimum is', min(Y_pred))
