###############
# Fetch Tweets#
###############


import time


# Twitter Auth Set up

import tweepy
from tweepy import OAuthHandler
 
consumer_key = 'RTKvdI1UV0zMPFNSTKNig'
consumer_secret = 'Yc1lZB3FUHg7lLDLfBPGveoLXlkLiD7Q4O0930EU'
access_token = '268391213-hekriVdzbAsaKfMy0UFHtbZu1KG9XUWJagdP3gPF'
access_secret = '0s4L2u0fKMqeL1Kd8bMsqMHeZaglDVaRwRRQVrFiqLg'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)



# Retrieve tweets

f = open('../data/SemEval13/dev/full/tweeter-dev-full-B.tsv','r');
out = open('../data/tweets2013_training.txt', 'w');
for line in f :
	time.sleep(6); # delays for 6 seconds to avoid reaching api rate limits
	if line != '':
		listLine = line.split('\t');

		try:
			status = api.get_status(listLine[0]);
			txt = status.text;
			txt = txt.replace('\n',' ');
			txt = txt.replace('\t',' ');
			out.write(status.text+'\t'+listLine[2]);
			print(status.text+' '+listLine[2]+'\n');
		except:
			print('Error');
