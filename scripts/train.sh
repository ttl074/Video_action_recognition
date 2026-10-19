#!/bin/bash
export PYTHONPATH=src
python scripts/run.py \
    --frame_dir /content/datasets/UCF50_frames_16 \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode 'train'