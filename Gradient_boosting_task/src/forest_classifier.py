import pandas as P
data_path = '.\\gbm-data.csv'
data = P.read_csv(data_path)

import numpy as np
X = np.matrix(data.values[:,1:])
Y = np.matrix(data.values[:,0])
Y = np.squeeze(np.asarray(Y))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=241)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

best_tree_num = 37 # Got it from previous task as minimal iteration number at learning rate 0.2
clf = RandomForestClassifier(n_estimators=best_tree_num, random_state=241)
clf.fit(X_train,Y_train)
Y_pred = clf.predict_proba(X_test)
loss = log_loss(Y_test,Y_pred)

print('Loss is %.2f' % loss)
