import pandas as P
data_path = '.\\scores.csv'
data = P.read_csv(data_path)

true = data['true'].values
logreg = data['score_logreg'].values
svm = data['score_svm'].values
knn = data['score_knn'].values
tree = data['score_tree'].values

# Get ROC-AUC score
from sklearn.metrics import roc_auc_score
print('ROC-AUC scores for:')
print('Logistic regression, SVM, KNN, decision tree')
print('%.2f %.2f %.2f %.2f' % 
      (roc_auc_score(true,logreg), roc_auc_score(true,svm), roc_auc_score(true,knn), roc_auc_score(true,tree)))

# Get precision-recall score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
plt.figure(1)

def precision_graph(data, subplot, max_precisions):
    recall_threshold = 0.7
    top_precision = []
    precision, recall, thresholds = precision_recall_curve(true,data)
    for i in range(len(recall)):
        if recall[i] > recall_threshold:
            top_precision.append(precision[i])
    max_precisions.append(max(top_precision))
    plt.subplot(subplot)
    plt.plot(range(len(precision)),precision)

max_precisions = []
# Logistic regression
precision_graph(logreg, 221, max_precisions)
# Support vector method
precision_graph(svm, 222, max_precisions)
# K nearest neighbors
precision_graph(knn, 223, max_precisions)
# Decision tree
precision_graph(tree, 224, max_precisions)
plt.show()

print('Max precisions among precision values with recall more than 70% for:')
print('Logistic regression, SVM, KNN, decision tree')
print(' '.join([str(x) for x in max_precisions]))
