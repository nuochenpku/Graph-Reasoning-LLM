
#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# local/xjsonfile/rftV2
# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
DPO_PATH=$3
SFT_PATH=$4
if [ "$OUTPUT" == "" ]; then
    OUTPUT=output/deepspeed/nlgreasoning/dpo_beta0.5/
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=25001 main.py  \
   --data_path local/jsonfile_graph/$DPO_PATH \
   --data_split 0,10,0 \
   --model_name_or_path $SFT_PATH \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 2048 \
   --learning_rate 5e-6  \
   --weight_decay 0. \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 100 \
   --seed 1234 \
   --beta 0.5 \
    --print_loss \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --data_output_path $OUTPUT \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log &
