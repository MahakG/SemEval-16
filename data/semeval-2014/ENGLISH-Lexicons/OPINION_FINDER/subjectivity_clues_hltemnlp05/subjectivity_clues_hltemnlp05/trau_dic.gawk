cat subjclueslen1-HLTEMNLP05.tff |gawk '{printf("%s\t%s\t%s\n",substr($3,7),substr($4,6),substr($6,15));}'>dic_english_with_POS.txt
cat dic_english_with_POS.txt |gawk '{print $1,$3}'|sort |uniq -c|gawk '{print $2,$3}'>dic_english_without_repetitions.txt 

