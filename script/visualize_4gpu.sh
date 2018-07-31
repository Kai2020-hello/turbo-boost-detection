#!/usr/bin/env bash

$DEVICE_ID=4,5,6,7
$config_file=configs/105/meta_105_quick_1_roipool.yaml

CUDA_VISIBLE_DEVICES=$DEVICE_ID python main.py \
    --device_id=$DEVICE_ID \
    --phase=visualize \
    --config_name=None \
    --debug=0 \
    --config_file=$config_file


