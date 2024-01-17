export MODEL_PATH='/cpfs/user/chennuo/CN/output/deepspeed/nlgreasoning/mistral_V1'
# mkdir -p $MODEL_PATH/$1
CUDA_VISIBLE_DEVICES=5,6,7 python3  evaluate_nlg.py --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 8 \
&> $MODEL_PATH/test_graph.log &