#!/bin/sh

DATASET=h36m
KEYPOINTS=detectron_custom_h36m
cp /docker_host/data/data_2d_${DATASET}_${KEYPOINTS}.npz /videopose3d/data

python run.py \
    --keypoints ${KEYPOINTS} \
    --dataset ${DATASET}
    -arc 3,3,3,3,3 \
    -c checkpoint \
    --evaluate d-pt-243.bin \
    --render \
    --viz-subject S1 \
    --viz-action Default \
    --viz-camera 0 \
    --viz-video /docker_host/data//videos/dance_02_8_secs.mp4 \
    --viz-output output.gif \
    --viz-size 3 \
    --viz-downsample 2 \
    --viz-limit 60
