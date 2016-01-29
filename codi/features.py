""" Count Hashtags """

import re;
def countHashtags(tweets):

	res = []
	for t in tweets :
		res.append([countHashtagsInTweet(t)])

	return res;


def countHashtagsInTweet(t):
	return len([m.start() for m in re.finditer('<ALMO>', t)])


""" All Caps: the number of words with all characters in upper case"""

def countUpperCaseWords(tweets) :
	res = []
	for t in tweets :
		res.append([countUpperCaseWordsInTweet(t)])
	return res

def countUpperCaseWordsInTweet(t):

	pattern = re.compile(r'^[A-Z\d]+$')

	words = t.split(" ")

	res =  0
	for w in words :

		if pattern.match(w) : 
			res += 1

	return res


""" Count Elongated Words """

def countElongated(tweets) :
	res = []
	for t in tweets :
		res.append([countElongatedInTweet(t)])
	return res

def countElongatedInTweet(t):
	return len([x for x in t.split() if re.search(r'((?i)[a-z!?])\1\1+', x)])



if __name__=='__main__':

	print(countElongatedInTweet("sooo toooooo mmmmmmm !!!!! ;) muhahahahaha")) 