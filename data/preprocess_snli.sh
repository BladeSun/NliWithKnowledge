#!/bin/bash


for F in ./snli_1.0*
do
	tmp_name=${F##*_}
	tmp_name=${tmp_name%.*}
	cat $F | awk -F "\t" '$1 != "-"' > tmp
	cat tmp | awk -F "\t" '{print $1}' > ${tmp_name}_label.tok
	cat tmp | awk -F "\t" '{print $6}' | perl tokenizer.perl -threads 5 -l 'en' > ${tmp_name}_h.tok
	cat tmp | awk -F "\t" '{print $7}' | perl tokenizer.perl -threads 5 -l 'en' > ${tmp_name}_t.tok
done

#turn lowercase
for F in ./*.tok
do
	cat $F | tr 'A-Z' 'a-z' > tmpfile	
	cat tmpfile > $F
done
