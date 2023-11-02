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
    python -u inference_v1.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0