#!/bin/bash
command=python
class=pageCluster.py
train=train
declare -a algo_array=("dbscan")
declare -a feature_array=("log-tf-idf" "tf-idf" )
declare -a data_array=("new_stackexchange" "rottentomatoes" "new_asp" "new_douban" "new_youtube" "new_tripadvisor" "new_hupu" "new_baidu")
for data in "${data_array[@]}"
do
	for algo in "${algo_array[@]}"
	do
		for feature in "${feature_array[@]}"
		do
			echo $command $class "$data" "$algo" "$feature" $train
			$command $class "$data" "$algo" "$feature" $train
		done
	done
done