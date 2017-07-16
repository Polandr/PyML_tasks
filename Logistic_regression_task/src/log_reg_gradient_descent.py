import numpy as np

def matrix_row(M, i):
    return np.squeeze(np.asarray(M[i,:]))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dot(x, y):
    if len(x) != len(y):
        return 0
    return sum([a * b for a, b in zip(x, y)])

def dst(x, y):
    if len(x) != len(y):
        return 0
    dst_vect = [a - b for a, b in zip(x, y)]
    return np.sqrt(dot(dst_vect,dst_vect))

# Gradient descent
# if C!=0 then L2-regularized regression
# else non-regularized regression
def gradient_descent(X, Y, k=0.1, C=0.):
    print('Starting gradient descent k=%.2f C=%.2f' % (k,C))
    eps = 1e-5
    max_iter = 10000
    n = X.shape[1]
    l = X.shape[0]
    w = [0] * n
    itr = 0
    while itr < max_iter:
        w_prev = list(w)
        for i in range(n):
            step = sum([X[j,i]*Y[j]*(1 - sigmoid(Y[j]*dot(w_prev, matrix_row(X, j)))) for j in range(X.shape[0])])
            w[i] = w_prev[i] + k/l * step - k*C*w_prev[i]
        itr += 1
        if dst(w,w_prev) < eps:
            break
    print('Finished in %d iterations' % itr)
    return w

# Read data
import pandas as P
data_path = 'D:\Programming\Machine Learning\LogisticRegression\\data-logistic.csv'
data = P.read_csv(data_path ,header=None, names=['Class','X','Y'])
X = np.matrix(data.dropna().as_matrix(columns=['X','Y']))
Y = np.matrix(data.dropna().as_matrix(columns=['Class']))
Y = np.squeeze(np.asarray(Y))

# Calculate weights and probabilities without regularization and with L2-regularization
w = gradient_descent(X,Y,C=0)
w_reg = gradient_descent(X,Y,C=10.)
Y_pred = [sigmoid(dot(w, matrix_row(X, i))) for i in range(X.shape[0])]
Y_reg_pred = [sigmoid(dot(w_reg, matrix_row(X, i))) for i in range(X.shape[0])]

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(Y, Y_pred)
auc_reg = roc_auc_score(Y, Y_reg_pred)
print('ROC-AUC score without regularization: %.3f' % auc)
print('ROC-AUC score with L2-regularization: %.3f' % auc_reg)
