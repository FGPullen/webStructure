#!/bin/bash
command=python
class=pageCluster.py
train=train
declare -a algo_array=("dbscan")
declare -a feature_array=("log-tf-idf")
declare -a date_array=("July30")
declare -a data_array=("asp" "youtube" "douban" "rottentomatoes" "hupu" "stackexchange")
for data in "${data_array[@]}"
do
    for date in "${date_array[@]}"
    do
        for algo in "${algo_array[@]}"
        do
            for feature in "${feature_array[@]}"
            do
                echo $command $class "$data" "$date" "$algo" "$feature" cv
                $command $class "$data" "$date" "$algo" "$feature" cv
            done
        done
	done
done