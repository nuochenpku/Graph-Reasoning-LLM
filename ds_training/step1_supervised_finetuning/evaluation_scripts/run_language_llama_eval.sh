export CUDA_VISIBLE_DEVICES=3
python prompt_lang_eval_math.py \
    --model_name_or_path_baseline xlingual/output_step1_llama2_7b_mixV1_4epoch_eos \
    --model_name_or_path_finetune xlingual/output_step1_llama2_7b_mixV1_4epoch_eos \
    --batch_size 2 \
&> xlingual/output_step1_llama2_7b_mixV1_4epoch_eos/test_bs2.log
    # xlingual/output_step1_llama2_7b_V1_10epoch
