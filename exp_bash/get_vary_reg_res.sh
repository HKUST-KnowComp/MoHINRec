#!/bin/bash
dt=$1
echo reg rmse mae
#for reg in 1e-06 1e-05 0.0001 0.001 0.01 0.05 0.1 0.5 1.0 10.0 100.0
#for reg in 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
#for reg in 0.01 0.02 0.03 0.04 0.05 0.06 0.1 0.5
#for reg in 0.07 0.08 0.09 0.2 0.3 0.4
#for reg in "1e-05" 0.0001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 1.0 10.0
#for reg in 0.2 0.3 0.32 0.34 0.36 0.38 0.4 1.0
#for reg in 0.001 0.01 0.05 0.1 0.2 0.4 0.5 0.6 0.8 1.0 2.0 4.0 6.0 8.0 10.0 100.0
for reg in 12.0 14.0 16.0 18.0 20.0 50.0
do
    #grep -n2 'fm_anova_kernel_glasso finish' log/"$dt"-200k_all_fm_non_con_reg_reg"$reg".log| grep -A 4 10000 |grep -E -o 'reg_W.+|avg rmse=.+'|grep -A 1 $reg | grep 'avg' | awk -F '[=,]' '{print $2,$4}' | awk '{t1 += $1; t2 += $2} END {printf("'$reg' %.4f %.4f\n", t1/NR, t2/NR)}'
    grep -n2 'fm_glasso_nn finish' log/"$dt"-50k_all_fm_nn_fm_reg_reg"$reg".log| grep -A 4 10000 |grep -E -o 'reg_W.+|avg rmse=.+'|grep -A 1 $reg | grep 'avg' | awk -F '[=,]' '{print $2,$4}' | awk '{t1 += $1; t2 += $2} END {printf("'$reg' %.4f %.4f\n", t1/NR, t2/NR)}'
    #grep -n2 'finish exp on all splits' log/"$dt"-200k_all_fm_non_con_reg_reg"$reg".log| grep -E -o 'reg_W.+|avg rmse=.+'| grep 'avg' | awk -F '[=,]' '{print $2,$4}' | awk '{t1 += $1; t2 += $2} END {printf("'$reg' %.4f %.4f\n", t1/NR, t2/NR)}'
done
