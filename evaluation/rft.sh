
export MODEL_PATH='/cpfs/user/chennuo/CN/output/graph_nlg/7b_v2'


for seed in $(seq 1 100)
do
    echo $seed
    mkdir -p $MODEL_PATH/$seed
    bash test_graph.sh $seed
    # CUDA_VISIBLE_DEVICES=4,5,6,7 python3  rft_llama.py --model_path $MODEL_PATH \
    # --streategy Parallel \
    # --batch_size 8 \
    # --seed $seed \
    # &> $MODEL_PATH/$seed/math_graph.log &

done
