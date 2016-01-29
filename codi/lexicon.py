""" Sentiment Dictionaries Module"""

import csv

""" Given a list of tweets and a path corresponding to the sentiment dictionaries file 
	returns a matrix with as many rows as tweets and five columns with the sentiments 
	of the tweet."""


def lexicon(tweets) :

	result = []

	dictio = readCsvFile('../data/polarityDictionaries/lexicon.csv');

	for t in tweets:
		result.append(lexiconTweet(t,dictio))

	return result



""" Given a tweet and the sentiment dictionary, traverse all the words within the tweet and add all the sentiments.
	Return a list with 5 elements one for each sentiment()"""

def lexiconTweet(tweet, dictio) :

	results = [0,0,0,0,0,0,0,0,0,0]

	tWords = tweet.split()

	for w in tWords :
		for i in range(10):
			if w in dictio:
				results[i] += int(dictio[w][i])

	return results
		


def readCsvFile(path) :
	result = {}
	with open(path,'r') as csvfile :
		reader = csv.reader(csvfile,delimiter=',',quotechar='"')
		for row in reader:
			features = []
			for i in range(23,33):
				features.append(row[i])
			result[row[0]] = features
	return result


if __name__ == '__main__':
	print(lexicon(["war fanatic bad fancy", "good behavour man do not cheat"]))


