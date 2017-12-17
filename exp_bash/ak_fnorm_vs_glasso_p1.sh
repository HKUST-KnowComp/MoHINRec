#!/bin/bash
python run_exp.py config/yelp_all.yaml -reg 0.0 &
python run_exp.py config/yelp_all.yaml -reg 1 &
python run_exp.py config/yelp_all.yaml -reg 10 &
python run_exp.py config/yelp_all.yaml -reg 50 &
python run_exp.py config/yelp_all.yaml -reg 100 &
