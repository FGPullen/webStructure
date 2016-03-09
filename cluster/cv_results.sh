#!/bin/bash
command=frameworkpython
class=pageCluster.py
cv=cv
declare -a algo_array=("kmeans")
<<<<<<< HEAD
declare -a feature_array=("log-tf-idf" "tf-idf" "binary")
declare -a data_array=("stackexchange" "zhihu" "rottentomatoes" "asp")
=======
declare -a feature_array=("tf-idf")
#declare -a data_array=("stackexchange" "zhihu" "rottentomatoes" "medhelp" "asp")
declare -a data_array=("zhihu")
>>>>>>> fd51a5ab17ef1a78eb8fdacf9eeb81a5ba0f1366
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
