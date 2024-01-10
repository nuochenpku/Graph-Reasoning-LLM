# export 
export MODEL_PATH='/cpfs/user/chennuo/CN/output/xmath/llama2_7b_en_sw_th'

CUDA_VISIBLE_DEVICES=0 python3  svamp_test.py --model_path $MODEL_PATH \
    --streategy Cross \
    --batch_size 32 \
&> $MODEL_PATH/svamp_cross_testbf16.log &

CUDA_VISIBLE_DEVICES=1 python3  generate_and_eval.py --model_path $MODEL_PATH \
    --streategy Cross \
    --batch_size 32 \
&> $MODEL_PATH/mgsm_cross_testbf16.log &


CUDA_VISIBLE_DEVICES=2 python3  svamp_test.py --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 32 \
&> $MODEL_PATH/svamp_parallel_testbf16.log &

CUDA_VISIBLE_DEVICES=3 python3  generate_and_eval.py --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 32 \
&> $MODEL_PATH/mgsm_Parallel_testbf16.log 

