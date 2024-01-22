export MODEL_PATH='/hpc2hdd/home/yli258/olddata/yuhanli/LLaMA_v2_ckpts/hf/Llama-2-13b-hf'
export SAVE_PATH='/hpc2hdd/home/yli258/jhaidata/Graph-Reasoning-LLM/datasets/probe_13b'

# mkdir -p $MODEL_PATH/$1
CUDA_VISIBLE_DEVICES=0,1,2,3 python3  graph_probing.py --model_path $MODEL_PATH \
    --streategy Parallel \
    --batch_size 6 \
    --save_dir $SAVE_PATH
# &> $MODEL_PATH/math_graph.log 