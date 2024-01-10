#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# local/xjsonfile/rftV2
# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2

if [ "$OUTPUT" == "" ]; then
    OUTPUT=/cpfs/user/chennuo/CN/output/Constrative_math/twostage_constrastive_13b/
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=29500 main.py  \
   --data_path local/jsonfile_verify/cpfs/user/chennuo/CN/XMATH-LLM/data/contrastive_trainV2.json \
   --data_split 10,0,0 \
   --model_name_or_path /cpfs/user/chennuo/CN/output/math/new_constrastive_13b/ \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 1024 \
   --learning_rate 2e-5  \
   --weight_decay 0. \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

#    9.65e-6
#    --offload \