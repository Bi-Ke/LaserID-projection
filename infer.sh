#!/bin/sh

exp_dir=eval
save_dir=checkpoints

INFER_CODE=infer_cenet_ppm_64x2048_official.py
config=configs/laserid_cenet_ppm_64x2048.yaml


now=$(date +"%Y%m%d_%H%M%S")
python ${exp_dir}/${INFER_CODE} --config=${config} 2>&1 | tee ${save_dir}/infer-$now.log

