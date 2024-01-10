#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./harry/output_step1_llama2_7b_dia_aff
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path local/harryjsonfile \
   --data_split 10,0,0 \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --per_device_train_batch_size 2  \
   --per_device_eval_batch_size 2 \
   --max_seq_len 700 \
   --learning_rate 9.65e-6  \
   --weight_decay 0. \
   --num_train_epochs 6  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

#    9.65e-6