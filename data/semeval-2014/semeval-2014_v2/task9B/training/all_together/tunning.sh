#!/bin/bash

dir=${1}
nu=${2}

folds=5
#
weka="java -Xmx12000m -Dfile.encoding=utf-8 -classpath \$CLASSPATH:/usr/share/java/weka.jar:/usr/share/java/libsvm.jar"
opts="weka.classifiers.functions.LibSVM"
param="-S 1 -K 2 -D 3 -G 0.0 -R 0.0 -N ${nu} -M 40.0 -C 1.0 -E 0.001 -P 0.1 -B"
for file in `ls ${dir}/*.arff.gz`
do
  dest="$(echo "${file}" | sed 's/.arff.gz//g')_"nu${nu}"_result.txt"
  cmm1="${weka} ${opts} ${param} -x "${folds}" -t ${file} -distribution -p 0"
  #echo ${file}
  #echo ${dest}
  #echo ${cmm1}
  eval ${cmm1} > ${dest}
done
