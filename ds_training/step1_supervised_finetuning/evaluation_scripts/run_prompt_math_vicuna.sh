#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=1
python prompt_eval_math.py \
    --model_name_or_path_baseline huggyllama/llama-7b \
    --model_name_or_path_finetune output_vic
