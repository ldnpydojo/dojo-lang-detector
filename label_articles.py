import json
import fileinput

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB

features = []
label_objects = {}
label_count = 0
labels   = []

#import data
with open('train_200.json', 'r') as f:
	for line in f:
		line = json.loads(line)
		features.append(line['text'])
		
		if line['lang'] not in label_objects.viewkeys():
			label_objects[line['lang']] = label_count
			label_count += 1

		labels.append(label_objects[line['lang']])

#vectorize
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(features)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

import pdb; pdb.set_trace()

#train cls
clf = MultinomialNB().fit(X_train_tfidf, labels)

predicted = clf.predict(X_train_tfidf)

for doc, lang in zip(features, predicted):
	print('%s => %s' % (doc, label_objects.keys()[label_objects.values().index(lang)]))

#for line in fileinput.input():
#	print line
