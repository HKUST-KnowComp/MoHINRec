from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
v = DictVectorizer()

D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
X = v.fit_transform(D)
print X

