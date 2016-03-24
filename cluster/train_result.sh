#!/bin/bash
command=python
class=pageCluster.py
train=train
declare -a algo_array=("dbscan")
declare -a feature_array=("tf-idf" "log-tf-idf")
#declare -a data_array=("medhelp")
declare -a data_array=("new_stackexchange" "new_rottentomatoes" "new_asp")
for data in "${data_array[@]}"
do
	for algo in "${algo_array[@]}"
	do
		for feature in "${feature_array[@]}"
		do
			$command $class "$data" "$algo" "$feature" $train
		done
	done
done