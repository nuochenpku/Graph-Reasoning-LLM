export CUDA_VISIBLE_DEVICES=2,3

python -m torch.distributed.launch --nproc_per_node=2  m_eval_math.py \
    --model_name_or_path_baseline xlingual/output_step1_llama2_7b_mixV1_10epoch_eos \
    --model_name_or_path_finetune xlingual/output_step1_llama2_7b_mixV1_10epoch_eos

# python -m torch.distributed.launch