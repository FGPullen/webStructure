#!/bin/bash
command=python
class=cluster_baselines.py
train=train
declare -a data_array=("asp" "youtube" "douban" "rottentomatoes" "hupu" "stackexchange")
for data in "${data_array[@]}"
do
    echo $command $class "$data"  cv
    #$command $class "$data"  cv
    $command $class "$data"  train

done