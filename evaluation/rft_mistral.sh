export MODEL_PATH='/cpfs/user/chennuo/CN/output/deepspeed/nlgreasoning/mistral_V1/'
# mkdir -p $MODEL_PATH/$1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3  rft_llama.py --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 16 \
&> $MODEL_PATH/math_graph_20_25.log &