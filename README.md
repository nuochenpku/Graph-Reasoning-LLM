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

## Trl Training

change Llama_path & evaluation file path.

```
python training/trl_demo.py
```