#!/bin/bash

off=2
on=1
dev=7

CUDA_VISIBLE_DEVICES=${dev} python sr_fine_tune.py --offline_env=${off} --online_env=${on} --data_type_indx=3 --seed=42
CUDA_VISIBLE_DEVICES=${dev} python sr_fine_tune.py --offline_env=${off} --online_env=${on} --data_type_indx=3 --seed=52
CUDA_VISIBLE_DEVICES=${dev} python sr_fine_tune.py --offline_env=${off} --online_env=${on} --data_type_indx=3 --seed=62
CUDA_VISIBLE_DEVICES=${dev} python sr_fine_tune.py --offline_env=${off} --online_env=${on} --data_type_indx=3 --seed=72