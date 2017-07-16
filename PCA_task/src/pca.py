import pandas as P
data_path = '.\\close_prices.csv'
dj_index_path = '.\\djia_index.csv'
data = P.read_csv(data_path, index_col='date')
dj_index = P.read_csv(dj_index_path, index_col='date')
X = data.as_matrix()

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_new = pca.fit_transform(X)

# Get the most principal component number (explaining 90% of variance)
variance_sum = 0
variance_threshold = 0.9
n_items = 0
for x in pca.explained_variance_ratio_:
    variance_sum += x
    n_items += 1
    if variance_sum >= variance_threshold:
        break
print('Explain component number: %d' % n_items)

# Get correlation between first component and DowJones Index
import numpy as np
first_component = np.squeeze(np.asarray(X_new[:,0]))
dj_index = np.squeeze(np.asarray(dj_index['^DJI']))
corrcoef = np.corrcoef(first_component,dj_index)[0,1]
print('Pearson correlation coefficient between first component and DowJones index: %.2f' % corrcoef)

# Get the most principal component
weights = [abs(x) for x in pca.components_[0]]
company_name = data.axes[1][weights.index(max(weights))]
print('The most significant company name is %s' % company_name)
# The ansewr is V the shortcut for Visa
