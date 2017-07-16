import pandas as P
data_path = '.\\gbm-data.csv'
data = P.read_csv(data_path)

import numpy as np
X = np.matrix(data.values[:,1:])
Y = np.matrix(data.values[:,0])
Y = np.squeeze(np.asarray(Y))

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=241)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
%matplotlib inline

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def collect_logloss(X,Y):
    logloss = []    
    for pred in clf.staged_decision_function(X):
        pred = np.squeeze(np.asarray(pred))
        Y_pred = [sigmoid(x) for x in pred]
        logloss.append(log_loss(Y,Y_pred))
    return logloss

rates = [1, 0.5, 0.3, 0.2, 0.1]
for rate in rates:
    
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=rate)
    clf.fit(X_train,Y_train)
    
    test_loss = collect_logloss(X_test, Y_test)
    train_loss = collect_logloss(X_train, Y_train)
    
    min_test_loss, min_test_loss_index = min(test_loss), test_loss.index(min(test_loss))
    min_train_loss, min_train_loss_index = min(train_loss), train_loss.index(min(train_loss))
    print('Learning rate: %.1f ; minimum test loss: %.2f at iteration %d ; minimum train loss: %.2f at iteration %d' % 
          (rate, min_test_loss, min_test_loss_index + 1, min_train_loss, min_train_loss_index + 1))
    
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
