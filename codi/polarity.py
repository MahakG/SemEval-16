"""Polarity Module"""




def countPolarity(tweets):
	"""
	 Count positive/negative words in a certain tweet from a list of tweets
	 
	 Parameters: list of tweets													
	 Return: a list of lists where the first element in the inner list 	
	  is a counter for the positive words and the second is 			
	  a counter for the negative words 	-> [[+,-],[+,-]...]			

	"""

	result = [];

	positiveWords = readTxtToList('../data/polarityDictionaries/positive-words.txt');
	posSet = set(positiveWords);	
	negativeWords = readTxtToList('../data/polarityDictionaries/negative-words.txt');
	negSet = set(negativeWords);
	for i,tweet in enumerate(tweets):
		result.append([0,0]);
		words = tweet.strip().split(' ');
		
		for w in words:
			if w.lower() in posSet:
				result[i][0]+= 1;
			if w.lower() in negSet:
				result[i][1]+= 1;

	return result;


#########################
#	UTILITY FUNCTIONS	#
#########################


def readTxtToList(path):
	"""
	 Read a file which contains the polarity words separated in different lines and return a list.
	 Parameters: path to file.
	 Return: A list with all the polarity words from the file.
	"""
	result = [];

	f = open(path);

	for line in f :
		if line != '' and line != '\n' and line[0] != ';':
			result.append(line.strip());

	return result;