#!/bin/bash
echo 'reg=0.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_regNone -n100 | grep 'avg'
echo 'reg=1.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_reg1.0 -n100 | grep 'avg'
echo 'reg=10.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_reg10.0 -n100 | grep 'avg'
echo 'reg=50.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_reg50.0 -n100 | grep 'avg'
echo 'reg=100.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_reg100.0 -n100 | grep 'avg'
echo 'reg=500.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_reg500.0 -n100 | grep 'avg'
echo 'reg=1000.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_reg1000.0 -n100 | grep 'avg'
echo 'reg=2000.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_reg2000.0 -n100 | grep 'avg'
echo 'reg=5000.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_reg5000.0 -n100 | grep 'avg'
echo 'reg=10000.0'
tail log/yelp-50k_all_fm_apk_fnorm_vs_glasso.log_reg1000.0 -n100 | grep 'avg'