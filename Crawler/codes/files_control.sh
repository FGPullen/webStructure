#! /bin/bash
test="test"
folder=../test_data/stackexchange/*
for FILE in $folder
do
    new_file=${FILE/stackexchange/$test}
    if test -e "$new_file"
    then
        echo $new_file
    else
        cp $FILE "../test_data/train/"
    fi
done
