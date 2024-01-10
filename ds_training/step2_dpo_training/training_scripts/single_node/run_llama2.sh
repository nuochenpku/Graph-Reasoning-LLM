#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# local/xjsonfile/rftV2
# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2

if [ "$OUTPUT" == "" ]; then
    OUTPUT=/cpfs/user/chennuo/CN/output/dpo_math/ds_7b_v1_reason_dpo_sft/
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2,3 --master_port=29592 main.py  \
   --data_path local/jsonfile_verify/cpfs/user/chennuo/CN/gsm8k-ScRel/data/dpo/train_ins_only_rea_dpo.json \
   --data_split 0,10,0 \
   --model_name_or_path /cpfs/user/chennuo/CN/output/dpo_math/ds_7b_rea_sft \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 768 \
   --learning_rate 2e-5  \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --beta 0.01 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --print_loss \
   --add_sft \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log &

#    9.65e-6
#    --gradient_checkpointing \