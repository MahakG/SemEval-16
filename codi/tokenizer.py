
#imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np
from scipy import sparse

import  nltk.util
import polarity
import skipGram

#############
# Tokenizer #
#############

#Retrieve training data

f = open('../data/tweets2013_train.txt','r');
tweets = [];
target = [];
for line in f :
	if line != '' and line != '\n':
		listLine = line.strip().split('\t');
		tweets.append(listLine[0]);
		target.append(listLine[1]);

#Retrieve testing data
f_test = open('../data/tweets2013_test.txt','r');
test_data = [];
test_target = [];
for line in f_test :
	if line != '' and line != '\n':
		listLine = line.strip().split('\t');
		test_data.append(listLine[0]);
		test_target.append(listLine[1]);

#Vectorization of tweets into an ocurrence matrix.
cv = CountVectorizer(stop_words='english',ngram_range=(1,1));
X_train_count = cv.fit_transform(tweets);
print(X_train_count.shape);

#Add columns for polarity dictionaries

polarityCols = polarity.countPolarity(tweets);
X_train_count = sparse.hstack(( X_train_count,polarityCols));
#print(X_train_count.shape);

#Add columns for skip-grams
skipGramDict = skipGram.generateSkipGramDict(tweets,1);
skipGramsCols = skipGram.generateSkipGramMatrix(len(tweets), skipGramDict);
print(len(skipGramsCols),len(skipGramsCols[0]));
X_train_count = sparse.hstack(( X_train_count, skipGramsCols));
print(X_train_count.shape);


#Transform ocurrences into frequencies.

tfidf = TfidfTransformer();
X_train_tfidf = tfidf.fit_transform(X_train_count);
print(X_train_tfidf.shape);



#Classifier
svm = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, n_iter=8, random_state=42);
svm.fit(X_train_tfidf, target);

#Vectorization of tweets into an ocurrence matrix.
X_test_count = cv.transform(test_data);

#Add columns for polarity dictionaries

polarityColsTest = polarity.countPolarity(test_data);
#print(newColumnsTest);
X_test_count = sparse.hstack(( X_test_count,polarityColsTest));

print(X_test_count.shape);

#Add columns for skip-grams

skipGramsColsTest =  skipGram.generateSkipGramMatrix(len(test_data),  skipGram.generateSkipGramTestDict(test_data,skipGramDict, 2));
print(len(skipGramsColsTest),len(skipGramsColsTest[0]));
X_test_count = sparse.hstack(( X_test_count, skipGramsColsTest));
print(X_test_count.shape);

#Transform ocurrences into frequencies.
X_test_tfidf = tfidf.transform(X_test_count);
#print(X_test_tfidf[4]);
print(X_test_tfidf.shape);
predicted = svm.predict(X_test_tfidf);

#Print results
#for doc, category in zip(test_data, predicted) :
#	print(doc+"\t"+category);


#Evaluation
print(np.mean(predicted == test_target));




		
