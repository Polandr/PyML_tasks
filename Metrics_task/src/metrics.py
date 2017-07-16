import pandas as P
data_path = '.\\classification.csv'
data = P.read_csv(data_path)

TP = FP = FN = TN = 0
for item in data.itertuples():
    if (item[1] == 1 and item[2] == 1):
        TP += 1
    if (item[1] == 1 and item[2] == 0):
        FN += 1
    if (item[1] == 0 and item[2] == 1):
        FP += 1
    if (item[1] == 0 and item[2] == 0):
        TN += 1
print('TP, FP, FN, TN')
print('%d %d %d %d' % (TP, FP, FN, TN))

Accuracy = (TP + TN) / (TP + FP + FN + TN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F_score = 2 * Precision * Recall / (Precision + Recall)
print('Accuracy, precision, recall, F-score')
print('%.2f %.2f %.2f %.2f' % (Accuracy, Precision, Recall, F_score))
