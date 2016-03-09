#!/bin/bash
command=python
class=pageCluster.py
cv=cv
declare -a algo_array=("kmeans" "wkmeans")
declare -a feature_array=("tf-idf" "binary")
declare -a data_array=("stackexchange" "zhihu" "rottentomatoes" "medhelp" "asp")
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