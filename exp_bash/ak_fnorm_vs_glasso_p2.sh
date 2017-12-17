#!/bin/bash
python run_exp.py config/yelp_all.yaml -reg 500 &
python run_exp.py config/yelp_all.yaml -reg 1000 &
python run_exp.py config/yelp_all.yaml -reg 2000 &
python run_exp.py config/yelp_all.yaml -reg 5000 &
python run_exp.py config/yelp_all.yaml -reg 10000 &
