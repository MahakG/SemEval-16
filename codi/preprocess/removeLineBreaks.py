# Remove Line Breaks

f = open('../data/tweets2013_testing.txt','r');
out = open('../data/tweets2013_test.txt', 'w');
before = '';
for line in f :
	if line != '':
		listLine = line.strip().split('\t');
		txt = before + listLine[0];
		txt = txt.replace('\n',' ');
		if len(listLine) == 1 :
			before = listLine[0];
		else :
			out.write(txt+'\t'+listLine[1]+'\n');
			before = '';
		