#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/cpfs/user/chennuo/CN/output/deepspeed/lora/output_step1_llama2_7b_lora
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT

deepspeed --include localhost:7 --master_port=29500 main.py  \
   --data_path local/jsonfile \
   --data_split 10,0,0 \
   --model_name_or_path /cpfs/shared/nlp/llama2/llama-2-7b-hf \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 2e-5  \
   --weight_decay 0. \
   --num_train_epochs 4  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --lora_dim 128 \
   --lora_module_name "mlp" \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

#    9.65e-6