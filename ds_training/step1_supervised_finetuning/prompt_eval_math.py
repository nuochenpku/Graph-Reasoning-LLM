# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import json
from pathlib import Path
import transformers
from typing import Optional, Dict, Sequence, List
from transformers import (
    AutoModelForCausalLM, )
import numpy as np
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=2,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help='Specify num of return sequences',
    )
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])

    args = parser.parse_args()

    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    # do_sample=False,
    #             temperature=0.0,
    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    # generate_ids = model.generate(inputs.input_ids,
    #                               do_sample=False,
    #                                 temperature=0.0,
    #                               max_new_tokens=256)
    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)
    # real_output_ids = [
    #         output_id[len(inputs.input_ids[i]) :] for i, output_id in enumerate(generate_ids)
    #     ]
    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    # result = tokenizer.batch_decode(real_output_ids,
    #                                 skip_special_tokens=True,
    #                                 clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()

import re
def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0

def prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts, labels):
    all_count = 0
    count = 0
    index = 0
    all_samples, corrects, wrongs = [], [], []
    # for i in range(0, len(prompts), 2):
    
    for prompt, label in zip(prompts, labels):
        # prompt = prompts[i:i+2]
        # label = labels[i:i+2]
        index += 1
        # print(prompt.type)
        # print(prompt)
        
        # inputs = tokenizer(prompt,  return_tensors="pt").to(device)
        inputs = tokenizer(prompt,  padding=True, return_tensors="pt").to(device)

        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                do_sample=True,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        # for pro, la in  zip(r_finetune_g, label):
            
            # ans = extract_last_num(pro)
            # result = dict()
            # # ans = extract_last_num(pro)
            # gd = extract_last_num(la)
            # print(pro)
            # print(la)
            # print('----------------')
        ans = extract_last_num(r_finetune_g[0])
        result = dict()
            # # ans = extract_last_num(pro)
        gd = extract_last_num(label)
            # gd = extract_last_num(la)
            # print(ans, gd)
        result['index'] = index
        result['input'] = r_finetune_g[0]
        # result['input'] = pro
        result['label'] = label
        # result['label'] = la
        result['pred_num'] = ans
        result['label_num'] = gd
        if ans == gd:
            count += 1
            corrects.append(result)
        else:
            wrongs.append(result)
        all_samples.append(result)
        all_count += 1
        print(count, all_count, count/all_count)
        # exit()



 
    print(count, all_count, count/all_count)
    
    # with open(Path(args.model_name_or_path_finetune) / "correct.json", "w", encoding='utf-8') as f:
    #     json.dump(corrects, f, ensure_ascii=False, indent=4)
    # with open(Path(args.model_name_or_path_finetune) / "wrong.json", "w", encoding='utf-8') as f:
    #     json.dump(wrongs, f, ensure_ascii=False, indent=4)
    # with open(Path(args.model_name_or_path_finetune) / "generate_all.json", "w", encoding='utf-8') as f:
    #     json.dump(all_samples, f, ensure_ascii=False, indent=4)
    print("====================prompt end=============================")
    print()
    print()


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def main():
    args = parse_args()

    

    device = torch.device("cuda:0")

    tokenizer = load_hf_tokenizer(args.model_name_or_path_finetune,
                                  fast_tokenizer=True)
    tokenizer.padding_side = 'left'
    # tokenizer.pad_token_id = 2
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = 0
    # tokenizer.pad_token_id = 0
#    model_baseline = create_hf_model(AutoModelForCausalLM,
#                                     args.model_name_or_path_baseline,
#                                     tokenizer, None)
    print('loading ckpt from ', args.model_name_or_path_finetune)
    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, None)


    # if tokenizer.pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    #         tokenizer=tokenizer,
    #         model=model_fintuned,
    #     )
#    model_baseline.to(device)
    model_fintuned.to(device)
    model_baseline = None
    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    labels = []
    prompts = []
    if args.language == "English":
        prompt_no_input = (
        "Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request in English. Please answer in English.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
        )
        fi = open("/vc_data/users/v-chennuo/CN/Math_reasoning/open_source/url-nlp/mgsm/mgsm_English.json", encoding="utf8")
        for line in fi:
            line = line.strip()
            o = json.loads(line)
            # prompts.append(o["prompt"])
            # labels.append(o["chosen"])
            prompt = prompt_no_input.format(query=o['query']) 
            prompts.append(prompt)
            labels.append(o["response"])
#        prompts = [
#                "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\t Answer: ",
#                "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\t Answer: ",
#                "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?\t Answer: ",
#                "Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?\t Answer: ",
#                "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?\t Answer: "
#        ]
    elif args.language == "Chinese":
        prompts = [
            "Human: 请用几句话介绍一下微软? Assistant:",
            "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
            "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
            "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
            "Human: 鸟类为什么要南迁过冬? Assistant:"
        ]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]

    from sklearn.utils import shuffle
    # prompts, labels = shuffle(np.array(prompts), np.array(labels))
    prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts, labels)


if __name__ == "__main__":
    main()
