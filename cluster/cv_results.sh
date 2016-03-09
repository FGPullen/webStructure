#!/bin/bash
command=python
class=pageCluster.py
cv=cv
declare -a algo_array=("kmeans")
declare -a feature_array=("log-tf-idf" "tf-idf" "binary")
declare -a data_array=("stackexchange" "zhihu" "rottentomatoes" "asp")
for data in "${data_array[@]}"
do
	for algo in "${algo_array[@]}"
	do
		for feature in "${feature_array[@]}"
		do
			$command $class "$data" "$algo" "$feature" $cv
		done
	done
done
