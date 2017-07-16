from sklearn import datasets
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
Y = newsgroups.target

from sklearn.svm import SVC
clf = SVC(C=1., kernel='linear', random_state=241)
clf.fit(X,Y)

row = abs(clf.coef_.toarray()[0].ravel())
top_ten_indicies = row.argsort()[-10:]
top_ten_values = row[row.argsort()[-10:]]
feature_mapping = vectorizer.get_feature_names()
names = [feature_mapping[idx] for idx in top_ten_indicies]
print('Top 10 significant words: ' + ' '.join(sorted(names)))
