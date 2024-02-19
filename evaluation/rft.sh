
export MODEL_PATH=$SFT_MODEL_PATH
# mkdir -p $MODEL_PATH/$1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3  rft_llama.py --model_path $MODEL_PATH \
    --batch_size 16 \
    --seed 20 \
&> $MODEL_PATH/rft_graph_20.log &
