for f in `ls models/${1}`
do
  echo ${f}
  cat ${f} | grep "+" | wc | gawk '{print $1}'
done
