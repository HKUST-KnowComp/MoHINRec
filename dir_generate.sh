#!/bin/bash
dt=$1
echo data type:  $dt
for N in 1 2 3 4 5
do
    pwd
    #cd ~/kdd17_src/
    cd /csproject/edbg/data/huan/kdd17_src
    cd data/$dt/exp_split/$N/
    #cd mf_features/path_count/
    #mv ratings_user.dat ratings_only_user.dat
    #mv ratings_item.dat ratings_only_item.dat
    ln -s ratings_train_"$N".txt ratings.txt
    mkdir "sim_res"
    mkdir "mf_features"
    cd mf_features
    mkdir "path_count"
    cd ../sim_res/
    mkdir "path_count"
done
