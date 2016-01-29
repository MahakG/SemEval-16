
#imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
#from sklearn.svm import LinearSVC as SGDClassifier


import numpy as np
from scipy import sparse

import nltk.util
import polarity
import skipGram
import features
import emoticons_ES
import twokenize_ES
import lexicon
import contexts

#############
# Tokenizer #
#############

num_pos = 0
num_neg = 0
num_neu = 0
#Retrieve training data

f = open('../data/train.txt','r');
tweets = [];
target = [];
for line in f :
	if line != '' and line != '\n':
		listLine = line.strip().split('\t');
		
		#Tokenize tweet
		listLine[0] = u" ".join(twokenize_ES.tokenize(listLine[0]))
		
		#Analize tweet
		listLine[0] = emoticons_ES.analyze_tweet(listLine[0])
		
		#RemovePunctuation
		#listLine[0] = u" ".join(twokenize_ES.remove_punct(listLine[0]))

		tweets.append(listLine[0]);
		target.append(listLine[1]);


#Retrieve testing data

f_test = open('../data/test.txt','r');
test_data = [];
test_target = [];
for line in f_test :
	if line != '' and line != '\n':
		listLine = line.strip().split('\t');

		#Tokenize tweet
		listLine[0] = u" ".join(twokenize_ES.tokenize(listLine[0]))

		#Analize tweet
		listLine[0]=emoticons_ES.analyze_tweet(listLine[0])

		#RemovePunctuation
		#listLine[0] = u" ".join(twokenize_ES.remove_punct(listLine[0]))

		test_data.append(listLine[0]);
		test_target.append(listLine[1]);

		if listLine[1] == 'positive' :
			num_pos+=1

		if listLine[1] == 'negative' :
			num_neg+=1

		if listLine[1] == 'neutral' :
			num_neu+=1

#Vectorization of tweets into an ocurrence matrix.
#cv = CountVectorizer(stop_words='english',ngram_range=(1,4));
cv = CountVectorizer(ngram_range=(1,2))
X_train_count = cv.fit_transform(tweets);
print(X_train_count.shape);

""" Get Contexts """
#contexts.getContexts(tweets)

#Add colums for amount of hashtags
"""
hashtagsCols = features.countHashtags(tweets)
print("hashtagsCols: ")
print(len(hashtagsCols),len(hashtagsCols[0]))
X_train_count = sparse.hstack(( X_train_count, hashtagsCols))
"""
#Add colums for amount of Upper Case Words
"""
upperCaseWordsCols = features.countUpperCaseWords(tweets)
print("upperCaseWordsCols: ")
print(len(upperCaseWordsCols),len(upperCaseWordsCols[0]))
X_train_count = sparse.hstack(( X_train_count, upperCaseWordsCols))
"""
#Add colums for amount of Elongated Words
"""
countElongatedCols = features.countElongated(tweets)
print("countElongatedCols: ")
print(len(countElongatedCols),len(countElongatedCols[0]))
X_train_count = sparse.hstack(( X_train_count, countElongatedCols))
"""
#Add columns for polarity dictionaries

polarityCols = polarity.countPolarity(tweets)
print("Polarity: ")
print(len(polarityCols),len(polarityCols[0]))
X_train_count = sparse.hstack(( X_train_count,polarityCols))



#Add columns for skip-grams
#skipGramDict = skipGram.getKMostFrequentSkipGrams(skipGram.generateSkipGramDict(tweets,1));
#skipGramDict = skipGram.generateSkipGramDict(tweets,1)
#print(tweets);
#skipGramsCols = skipGram.generateSkipGramMatrix(len(tweets), skipGramDict)
#print("skipGram: ")
#print(len(skipGramsCols),len(skipGramsCols[0]))
#print(skipGramDict);
#print(skipGramsCols);
#X_train_count = sparse.hstack(( X_train_count, skipGramsCols))
#X_train_count = sparse.hstack(( polarityCols, skipGramsCols));
#print(X_train_count.shape);

#Add colums for lexicons
"""
lexiconCols = lexicon.lexicon(tweets)
print("lexiconCols: ")
print(len(lexiconCols),len(lexiconCols[0]))
X_train_count = sparse.hstack(( X_train_count, lexiconCols))
"""

#Transform ocurrences into frequencies.

tfidf = TfidfTransformer();

X_train_tfidf = tfidf.fit_transform(X_train_count);

print(X_train_tfidf.shape);

#Classifier
#svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=8, random_state=42);
svm = SGDClassifier()
svm.fit(X_train_tfidf, target);

#Vectorization of test_data into an ocurrence matrix.
X_test_count = cv.transform(test_data);


#Add columns for polarity dictionaries

polarityColsTest = polarity.countPolarity(test_data);
#print(newColumnsTest);
X_test_count = sparse.hstack(( X_test_count,polarityColsTest));

print(X_test_count.shape);


#Add columns for skip-grams
#skipGramTestDict = skipGram.generateSkipGramTestDict(test_data, skipGramDict, 2);
#print(test_data);
#print(skipGramTestDict)
#skipGramsColsTest =  skipGram.generateSkipGramMatrix(len(test_data), skipGramTestDict);
#print(len(skipGramsColsTest),len(skipGramsColsTest[0]));
#print(skipGramsColsTest);
#X_test_count = sparse.hstack(( X_test_count, skipGramsColsTest));
#print(X_test_count.shape);


#Add colums for amount of hashtags
#hashtagsColsTest = features.countHashtags(test_data)
#print("hashtagsColsTest: ")
#print(len(hashtagsColsTest),1)

#X_test_count = sparse.hstack(( X_test_count, hashtagsColsTest))


#Add colums for amount of Upper Case Words
"""
upperCaseWordsColsTest = features.countHashtags(test_data)
print("upperCaseWordsColsTest: ")
print(len(upperCaseWordsColsTest),len(upperCaseWordsColsTest[0]))

X_test_count = sparse.hstack(( X_test_count, upperCaseWordsColsTest))
"""

#Add colums for amount of Elongated Words
"""
countElongatedColsTest = features.countElongated(test_data)
print("countElongatedColsTest: ")
print(len(countElongatedColsTest),len(countElongatedColsTest[0]))

X_test_count = sparse.hstack(( X_test_count, countElongatedColsTest))
"""

#Add columns for amount of lexicon
"""
lexiconColsTest = lexicon.lexicon(test_data)
print("lexiconColsTest: ")
print(len(lexiconColsTest),len(lexiconColsTest[0]))

X_test_count = sparse.hstack((X_test_count,lexiconColsTest))
"""


#Transform ocurrences into frequencies.

X_test_tfidf = tfidf.transform(X_test_count);
#print(X_test_tfidf[4]);
print(X_test_tfidf.shape);


predicted = svm.predict(X_test_tfidf);

#Print results
#for doc, category in zip(test_data, predicted) :
#	print(doc+"\t"+category);


#Evaluation
ok_pos = 0
predicted_pos = 0

ok_neg = 0
predicted_neg = 0

ok_neu = 0
predicted_neu = 0
for i,x in enumerate(predicted) :
	if test_target[i] == 'negative' and predicted[i] != test_target[i] and i <500:
		print("Bad: "+test_data[i]+"\t"+predicted[i]+"\n")
		print(X_test_count.getrow(i),"\n")
	if x == 'positive' and predicted[i] == test_target[i] :
		ok_pos += 1

	if x=='positive': 
		predicted_pos += 1

	if x == 'negative' and predicted[i] == test_target[i] :
		print("Ok: "+test_data[i]+"\t"+predicted[i]+"\n")
		print(X_test_count.getrow(i),"\n")
		ok_neg += 1

	if x=='negative': 
		predicted_neg += 1

	if x=='neutral' and predicted[i] == test_target[i]: 
		ok_neu += 1

	if x=='neutral': 
		predicted_neu += 1


print(ok_pos)
print(predicted_pos)
print(num_pos)

print(ok_neg)
print(predicted_neg)
print(num_neg)

print(ok_neu)
print(predicted_neu)
print(num_neu)

print("Aciertos: ",(ok_pos+ok_neg+ok_neu))
print("Suma: ",(num_pos + num_neg + num_neu))
print("Suma: ",(predicted_pos + predicted_neg + predicted_neu))

print("Accuracy: ",(ok_pos+ok_neg+ok_neu)/(predicted_pos + predicted_neg + predicted_neu))

precision_pos = ok_pos /predicted_pos 
recall_pos = ok_pos /num_pos

precision_neg = ok_neg / predicted_neg
recall_neg = ok_neg / num_neg

precision_neu = ok_neu / predicted_neu
recall_neu = ok_neu / num_neu

f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) 
f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)

f1_neu = 2 * precision_neu * recall_neu / (precision_neu + recall_neu)

print("F1 Pos: ",f1_pos)
print("F1 Neg: ",f1_neg)
print("F1 Neu: ",f1_neu)

score = (f1_pos + f1_neg) / 2

print(score)




		
