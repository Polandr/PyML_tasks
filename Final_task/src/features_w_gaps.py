import pandas
data_path = '.\\features.csv'
data = pandas.read_csv(data_path, index_col='match_id')

# Удаление признаков, связанных с итогами матча
X = data.drop(
    labels=['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
            'barracks_status_radiant', 'barracks_status_dire']
    , axis=1)

# Вычленение признаков, в которых есть пропуски
non_null = X.count()
print('Features with gaps:')
print(non_null.where(non_null < X.shape[0]).dropna())
