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
from transformers import (
    AutoModelForCausalLM, )
import numpy as np
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer

logger = logging.getLogger(__name__)


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
        default=1,
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
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)

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

    generate_ids = model.generate(inputs.input_ids,
                                  attention_mask = inputs.attention_mask,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

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
                prompts, labels, lang):
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
        
        inputs = tokenizer(prompt,  return_tensors="pt").to(device)

        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                do_sample=True,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        ans = extract_last_num(r_finetune_g[0])
        result = dict()
        # ans = extract_last_num(pro)
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
    
    
    # with open(Path(args.model_name_or_path_finetune) / f"{lang}_correct.json", "w", encoding='utf-8') as f:
    #     json.dump(corrects, f, ensure_ascii=False, indent=4)
    # with open(Path(args.model_name_or_path_finetune) / f"{lang}_wrong.json", "w", encoding='utf-8') as f:
    #     json.dump(wrongs, f, ensure_ascii=False, indent=4)
    # with open(Path(args.model_name_or_path_finetune) / f"{lang}_generate_all.json", "w", encoding='utf-8') as f:
    #     json.dump(all_samples, f, ensure_ascii=False, indent=4)
    print(lang,': ')
    print(count, all_count, count/all_count)
    print("====================prompt end=============================")
    print()
    print()
    return count/all_count


def batch_prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts, labels, lang):
    all_count = 0
    count = 0
    index = 0
    all_samples, corrects, wrongs = [], [], []
    # for i in range(0, len(prompts), 2):
    for i in range(0, len(prompts), args.batch_size):
    
    
    # for prompt, label in zip(prompts, labels):
        # prompt = prompts[i:i+2]
        # label = labels[i:i+2]
        index += 1
        # print(prompt.type)
        # print(prompt)
        prompt = prompts[i: i+ args.batch_size]
        labels_ = labels[i: i+ args.batch_size]
        
        inputs = tokenizer(prompt,  return_tensors="pt", padding=True).to(device)

        outputs = generate(model_fintuned,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                do_sample=True,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        
        for output, label in zip(outputs, labels_):
            
        
            ans = extract_last_num(output)
            result = dict()
        # ans = extract_last_num(pro)
            gd = extract_last_num(label)
            result['index'] = index
            result['input'] = output
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
    
    
    # with open(Path(args.model_name_or_path_finetune) / f"{lang}_correct.json", "w", encoding='utf-8') as f:
    #     json.dump(corrects, f, ensure_ascii=False, indent=4)
    # with open(Path(args.model_name_or_path_finetune) / f"{lang}_wrong.json", "w", encoding='utf-8') as f:
    #     json.dump(wrongs, f, ensure_ascii=False, indent=4)
    # with open(Path(args.model_name_or_path_finetune) / f"{lang}_generate_all.json", "w", encoding='utf-8') as f:
    #     json.dump(all_samples, f, ensure_ascii=False, indent=4)
    print(lang,': ')
    print(count, all_count, count/all_count)
    print("====================prompt end=============================")
    print()
    print()
    return count/all_count


def main():
    args = parse_args()

    

    device = torch.device("cuda:0")

    tokenizer = load_hf_tokenizer(args.model_name_or_path_finetune,
                                  fast_tokenizer=True)
    # padding_side = 'right'
    tokenizer.pad_token_id = 2
#    model_baseline = create_hf_model(AutoModelForCausalLM,
#                                     args.model_name_or_path_baseline,
#                                     tokenizer, None)
    print('loading ckpt from ', args.model_name_or_path_finetune)
    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, bf16=True)

#    model_baseline.to(device)
    model_fintuned.to(device)
    model_baseline = None
    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    
    
    # langs = [ 'German', 'Spanish', 'Chinese','English']
    results = {}
    langs = ['English','Chinese','Bengali', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'Swahili', 'Telugu', 'Thai', 'En_gsm8k']
    for lang in langs:
        
        
        if lang ==  'En_gsm8k':
            prompt_no_input = (
            "Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request in English. Please answer in English.\n\n"
            "### Instruction:\n{query}\n\n### Response:"
            )
            # lang = 'English'
            eval_path = f"/vc_data/users/v-chennuo/CN/Math_reasoning/open_source/gsm8k-ScRel/data/test_use.jsonl"
        else:
            eval_path = f"/vc_data/users/v-chennuo/CN/Math_reasoning/open_source/url-nlp/mgsm/mgsm_{lang}.json"
        
            prompt_no_input = (
            "Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request in {lang}. Please answer in {lang}.\n\n"
            "### Instruction:\n{query}\n\n### Response:"
            )
        
        labels = []
        prompts = []
        fi = open(eval_path, encoding="utf8")
        for line in fi:
            line = line.strip()
            o = json.loads(line)
            prompt = prompt_no_input.format(query=o['query']) 
            prompts.append(prompt)
            labels.append(o["response"])
   

        from sklearn.utils import shuffle
        # prompts, labels = shuffle(np.array(prompts), np.array(labels))
        if args.batch_size == 1:
            result = prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                    prompts, labels, lang)
        else:
            result = batch_prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                    prompts, labels, lang)
        results[lang] = result
        
    average = sum(results.values()) / len(results)
    import csv
    with open(Path(args.model_name_or_path_finetune) / f"MSGM_evaluate_results_bs{args.batch_size}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Language', 'Accuracy'])
        for key, value in results.items():
            writer.writerow([key, value])
        writer.writerow(['Average', average])

    


if __name__ == "__main__":
    main()
