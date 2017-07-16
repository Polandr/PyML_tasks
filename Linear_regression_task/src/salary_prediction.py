import pandas as P
data_train_path = '.\\salary-train.csv'
data_test_path = '.\\salary-test-mini.csv'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
import numpy as np
tfidf = TfidfVectorizer(min_df=5)
encoder = DictVectorizer()

def prepare_data(data_path, data_type='train', vectorizer=tfidf, enc=encoder):
    # Read data from file
    data = P.read_csv(data_path)
    
    # Transform text to lower case
    data['FullDescription'] = data['FullDescription'].str.lower()
    data['LocationNormalized'] = data['LocationNormalized'].str.lower()
    data['ContractTime'] = data['ContractTime'].str.lower()
    
    # Erase non-alphabetical and non-number symbols and fill gaps
    data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)    
    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    
    # TF-IDF and one-hot encoding
    if (data_type == 'train'):
        X_description = vectorizer.fit_transform(data['FullDescription'])
        X_location_contract = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    elif (data_type == 'test'):
        X_description = vectorizer.transform(data['FullDescription'])
        X_location_contract = enc.transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    else:
        print('Undefined data type')
        return data
    
    # Concatenate data
    X = hstack([X_description,X_location_contract])
    
    if (data_type == 'train'):
        # Extract target variable vector
        Y = np.matrix(data.dropna().as_matrix(columns=['SalaryNormalized']))
        Y = np.squeeze(np.asarray(Y))
        return X, Y
    elif (data_type == 'test'):
        return X

X_train, Y_train = prepare_data(data_train_path, 'train')
X_test = prepare_data(data_test_path, 'test')

from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0, random_state=241)
clf.fit(X_train, Y_train)
Y_test = clf.predict(X_test)

print('Predicted salaries:')
print(' '.join([str(x) for x in Y_test]))
