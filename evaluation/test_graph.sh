export MODEL_PATH=$1
# mkdir -p $MODEL_PATH/$1
CUDA_VISIBLE_DEVICES=6,7 python3  evaluate_nlg.py --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 6 \
&> $MODEL_PATH/test_graph.log &
