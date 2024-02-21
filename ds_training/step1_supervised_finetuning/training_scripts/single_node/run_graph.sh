
#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
DATA_PATH=$3
MODEL_PATH=$4
if [ "$OUTPUT" == "" ]; then
    OUTPUT=/output/deepspeed/nlgreasoning/
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi 
mkdir -p $OUTPUT

deepspeed --include localhost:0,1,2,3 --master_port=25001 main.py  \
   --data_path local/jsonfile_graph/$DATA_PATH \
   --data_split 10,0,0 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 2048 \
   --learning_rate 5e-6  \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 500 \
   --seed 1234 \
   --save_interval 5000 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --data_output_path $OUTPUT \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log &
