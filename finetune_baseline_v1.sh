#!/bin/bash
set -x

PARTITION=llm4
JOB_NAME=minigpt4
GPUS_PER_NODE=1

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    --quotatype=reserved \
    torchrun --nproc-per-node 1 train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml