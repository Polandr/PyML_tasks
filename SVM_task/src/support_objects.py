import pandas as P
data_path = '.\\svm-data.csv'
data = P.read_csv(data_path, header=None, names=['Class','X','Y'])

import numpy as np
X = np.matrix(data.dropna().as_matrix(columns=['X','Y']))
Y = np.matrix(data.dropna().as_matrix(columns=['Class']))
Y = np.squeeze(np.asarray(Y))

from sklearn.svm import SVC
clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, Y)
print('Support object numbers: ' + ' '.join([str(x) for x in clf.support_]))
print('Support vectors:')
for idx in clf.support_:
    print(str(np.squeeze(np.asarray(X[idx]))) + ' : ' + str(Y[idx]))
