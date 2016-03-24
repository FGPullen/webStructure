#!/bin/bash
command=python
class=pageCluster.py
cv=cv
#cv=train
declare -a algo_array=("dbscan" "kmeans")
declare -a feature_array=("log-tf-idf" "tf-idf")
declare -a data_array=("new_stackexchange" "new_rottentomatoes")
#declare -a data_array=("zhihu")
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
