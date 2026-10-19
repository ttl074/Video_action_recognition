#!/bin/bash
export PYTHONPATH=src
python scripts/run.py \
    --ckpt "${1:?usage: test.sh <checkpoint_path>}" \
    --model_type lrcn \
    --n_classes 50 \
    --batch_size 4 \
    --mode eval
