#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
python prompt_eval_math.py \
    --model_name_or_path_baseline xlingual/output_step1_llama2_7b_mixV1_10epoch_eos \
    --model_name_or_path_finetune xlingual/output_step1_llama2_7b_mixV1_10epoch_eos
