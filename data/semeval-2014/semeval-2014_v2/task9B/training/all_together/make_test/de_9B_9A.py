

import sys

taskb_result_file = sys.argv[1]
taska_input_file = sys.argv[2]

with open(taskb_result_file) as fh:
    result_dic = dict([(b, c) for a, b, c in [line.split() for line in fh.read().split('\n') if line]])

with open(taska_input_file) as fh:
    list = ["\t".join((a, b, c, d, result_dic[b]))  for a, b, c, d, e in [line.split() for line in fh.read().split('\n') if line]]
    
print "\n".join(list)
