#!/bin/bash

#WEKA
weka="java -Xmx12000m -Dfile.encoding=utf-8 -classpath \$CLASSPATH:/usr/share/java/weka.jar:/usr/share/java/libsvm.jar"
opts="weka.classifiers.functions.LibSVM"
param="-S 1 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.45 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -B"

prefix=${1}
for training in `ls ${prefix}/*training*.arff.gz`
do
  test="$(echo "${training}" | sed 's/_training./_test./g')"
  model="$(echo "${training}" | sed 's/_training.arff.gz/.model/g')"
  result="$(echo "${training}" | sed 's/_training.arff.gz/.result.txt/g')"
  cmm1="${weka} ${opts} ${param} -t ${training} -T ${test} -d ${model}"
  cmm2="${weka} ${opts} -l ${model} -T ${test} -distribution -p 0"
  echo $training
  echo $test
  echo $model
  echo $result
  echo $cmm1
  echo $cmm2
  echo
  eval $cmm1
  eval ${cmm2} > ${result}
done

