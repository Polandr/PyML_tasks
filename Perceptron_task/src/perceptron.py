import pandas as P
train_path = '.\\perceptron-train.csv'
test_path = '.\\perceptron-test.csv'
train_data = P.read_csv(train_path,header=None, names=['Class','X','Y'])
test_data = P.read_csv(test_path,header=None, names=['Class','X','Y'])

# Get data
import numpy as np
X_train = np.matrix(train_data.dropna().as_matrix(columns=['X','Y']))
Y_train = np.matrix(train_data.dropna().as_matrix(columns=['Class']))
Y_train = np.squeeze(np.asarray(Y_train))
X_test = np.matrix(test_data.dropna().as_matrix(columns=['X','Y']))
Y_test = np.matrix(test_data.dropna().as_matrix(columns=['Class']))
Y_test = np.squeeze(np.asarray(Y_test))

from sklearn.linear_model import Perceptron
perceptron = Perceptron(random_state=241)
from sklearn.metrics import accuracy_score

# Learning without normalization:
perceptron.fit(X_train, Y_train)
predictions = perceptron.predict(X_test)
score_unnormalized = accuracy_score(Y_test, predictions)

# Normalization:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Learning with normalization
perceptron.fit(X_train_scaled, Y_train)
predictions = perceptron.predict(X_test_scaled)
score_normalized = accuracy_score(Y_test, predictions)

print('Normalized score : %.3f ,unnormalized score: %.3f ,difference: %.3f' % 
      (score_unnormalized, score_normalized, (score_normalized-score_unnormalized)))
