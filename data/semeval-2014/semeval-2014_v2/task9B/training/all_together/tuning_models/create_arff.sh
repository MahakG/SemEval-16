python ../../../../../../from_freeling_to_weka.py -l -m 1 -W ../lexicons/AFINN-111_dic.txt ../all2013_joined2_CMU.txt ../all2013_info.txt AFINN_joined2_m1
python ../../../../../../from_freeling_to_weka.py -l -m 2 -W ../lexicons/AFINN-111_dic.txt ../all2013_joined2_CMU.txt ../all2013_info.txt AFINN_joined2_m2
python ../../../../../../from_freeling_to_weka.py -l -m 1 -W ../lexicons/AFINN-111_dic.txt -G ../lexicons/AFINN-111.txt  ../all2013_joined2_CMU.txt ../all2013_info.txt AFINN_joined2_global_m1
python ../../../../../../from_freeling_to_weka.py -l -m 2 -W ../lexicons/AFINN-111_dic.txt -G ../lexicons/AFINN-111.txt  ../all2013_joined2_CMU.txt ../all2013_info.txt AFINN_joined2_global_m2
python ../../../../../../from_freeling_to_weka.py -l -m 1 -W ../lexicons/AFINN-111_dic.txt -G ../lexicons/AFINN-111.txt  ../all2013_CMU.txt ../all2013_info.txt AFINN_global_m1
python ../../../../../../from_freeling_to_weka.py -l -m 2 -W ../lexicons/AFINN-111_dic.txt -G ../lexicons/AFINN-111.txt  ../all2013_CMU.txt ../all2013_info.txt AFINN_global_m2



