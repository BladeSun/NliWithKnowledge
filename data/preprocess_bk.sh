#!/bin/bash


for F in ./*_syn.txt
do
	tmp_name=${F%.*}
	cat $F | awk -F "\t" '$1 != "-"' > tmp
	cat tmp | awk -F "\t" '{print $2}' > ${tmp_name}_h.syntok
	cat tmp | awk -F "\t" '{print $3}' > ${tmp_name}_t.syntok
done
