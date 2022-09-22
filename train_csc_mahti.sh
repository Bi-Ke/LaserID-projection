#!/bin/sh
exp_dir=train

TRAIN_CODE=train_laserid_cenet_ppm_64x2048.py
config=configs/laserid_cenet_ppm_64x2048.yaml


save_dir=checkpoints
mkdir -p ${save_dir}

now=$(date +"%Y%m%d_%H%M%S")
python ${exp_dir}/${TRAIN_CODE} --config=${config} 2>&1 | tee ${save_dir}/train-$now.log

