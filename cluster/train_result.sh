#!/bin/bash
command=python
class=pageCluster.py
train=train
declare -a algo_array=("dbscan")
declare -a feature_array=("log-tf-idf")
declare -a date_array=("July30")
declare -a data_array=("rottentomatoes" "youtube" "douban" "huffingtonpost" "tripadvisor" "hupu" "baidu" "photozo")
for data in "${data_array[@]}"
do
    for date in "${date_array[@]}"
    do
        for algo in "${algo_array[@]}"
        do
            for feature in "${feature_array[@]}"
            do
                echo $command $class "$data" "$date" "$algo" "$feature" $train
                $command $class "$data" "$date" "$algo" "$feature" $train
            done
        done
	done
done