# Graph-Reasoning-LLM
this is project for training explicit graph reasoning large language models.

## Requirements

```
pip -r install requirements.txt
```

## Generate all train set

```
cd scripts
bash generate_all_train_datasets.sh
```

## Generate all test set

```
cd scripts
bash generate_all_test_datasets.sh
```

## Trl Training

change Llama_path & evaluation file path.

```
# dpo
python training/dpo.py 
# sft
python training/sft.py
```
