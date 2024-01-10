export CUDA_VISIBLE_DEVICES=1
python harrry_eval.py \
    --model_name_or_path_baseline /vc_data/users/v-chennuo/CN/rl4reasoning/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/harry/output_step1_llama2_7b_dia_aff \
    --model_name_or_path_finetune /vc_data/users/v-chennuo/CN/rl4reasoning/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/harry/output_step1_llama2_7b_dia_aff