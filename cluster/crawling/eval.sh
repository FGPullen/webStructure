#!/bin/bash
command=python
class=eval_crawl.py
train=train
declare -a algo_array=("0" "1")
declare -a size_array=("5001")
declare -a date_array=("July30")
declare -a data_array=("asp" "youtube" "douban" "rottentomatoes" "hupu" "stackexchange")
for data in "${data_array[@]}"
do
    for date in "${date_array[@]}"
    do
        for algo in "${algo_array[@]}"
        do
            for feature in "${size_array[@]}"
            do
                echo $command $class "$data" "$date" "$algo" "$feature" target
                $command $class "$data" "$date" "$algo" "$feature" target
            done
        done
	done
done