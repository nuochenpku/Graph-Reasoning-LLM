import json
import re
from pathlib import Path
from typing import Callable
import random
import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Sequence, List
import argparse


CHOICES = ['A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0
    
    
def check(key, truth, predict):
    
    if key in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
        if '###' in predict:
            if 'yes' in truth.lower() and 'yes' in predict.split('###')[-1].lower():
                # correct_samples[key].append(v)
                return True
            elif 'no' in truth.lower() and 'no' in predict.split('###')[-1].lower():
                return True
            return False
        else:
            matches = re.findall(r'\b(yes|no)\b', predict, flags=re.IGNORECASE)
            if matches:
                last_match = matches[-1].lower()
                if last_match == 'yes' and 'yes' in truth.lower():
                    return True
                elif last_match == 'no' and 'no' in truth.lower():
                    return True
            else:
                return False
    elif key in ['flow', 'shortest', 'triplet']:
      
        t_num = extract_last_num(truth)
        p_num = extract_last_num(predict.split('###')[-1])
        if abs(t_num - p_num) < 1e-2:
            return True
        return False
                
    elif key == 'topology':
        
        # elif key == 'topology':
        
        if '###' in predict:
            pre = predict.split('###')[-1].strip(' ')
            truth = truth.split('###')[-1].strip(' ')
            if truth in pre or pre in truth:
                return True
            return False
        else:
            truth = truth.split('###')[-1].split(',')
            for t in truth:
                if t in predict or t.strip(' ') in predict:
                    return True
            return False



def main(
    args,
    gsm8k_test_jsonl: str = "./data/mgsm",
    is_bf16: bool = True,
    save_dir: str  = None,
):
    batch_size = args.batch_size
    print(f"main start, is_bf16:{is_bf16}, batch_size:{batch_size}")
    
    model_path = args.model_path
    model, tokenizer = get_model(model_path, is_bf16=is_bf16)
    print("model loaded")

    batch_llama = get_batch_llama(model, tokenizer, args)

    pure_model = model_path.split('/')[-1]
    if save_dir is None:
        save_dir = f"/Graph_RFT_Data/Results/{pure_model}/{model.dtype}/rft"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    

    tasks = ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite', 'flow', 'shortest', 'triplet', 'topology']
    sources = []
    targets = []
    results = {}
    for seed in  range(0,seed):
        print(f'===========we are testing in {seed}====================')
        for lang in tasks:
            print(f'===========we are testing in {lang}====================')
            
            with open(f'args.data_path/{lang}_train.json') as f:
                datas = f.readlines()
                
            gen_datas_jsonl = Path(save_dir) / f"{seed}_gen_{lang}_datas.jsonl"
            start_index = (
                len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
            )
            print(f"start_index: {start_index}")
            
            gsm8k_datas = [json.loads(item) for item in datas]
            # for key, value in datas.items():
            #     gsm8k_datas.append(value)
            
            
            for i in tqdm(range(start_index, len(gsm8k_datas), batch_size)):
                cur_gsm8k_batch = gsm8k_datas[i : i + batch_size]
                input_str_list, output_str_list = gsm8k_batch_gen(lang, 
                    cur_gsm8k_batch, batch_llama, args
                )
                for j, (gsm8k_data, input_str, output_str) in enumerate(
                    zip(cur_gsm8k_batch, input_str_list, output_str_list)
                ):
                    with open(gen_datas_jsonl, "a") as f:
                        json.dump(
                            dict(
                                index=i + j,
                                source_data=gsm8k_data,
                                input_str=input_str,
                                output_str=output_str,
                                task=lang
                            ),
                            f,
                        )
                        f.write("\n")

        # calculate acc
            with open(gen_datas_jsonl) as f:
                gen_datas = [json.loads(line) for line in f]

            correct_results = []
            wrong_results = []
            for gen in gen_datas:
                result = dict(
                    **gen,
                    extract_true=gen["source_data"]["answer"],
                    extract_pred=gen["output_str"].lstrip(),
                    is_correct=None,
                )
                
                if check(lang, result["extract_true"].lower(), result["extract_pred"].lower()):
                
                # if result["extract_pred"].lower() in result["extract_true"].lower() or result["extract_true"].lower() in result["extract_pred"].lower():
                    result["is_correct"] = True
                    correct_results.append(result)
                else:
                    result["is_correct"] = False
                    wrong_results.append(result)

        #     print(f'=======done {lang}============')
            result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results)/(len(correct_results) + len(wrong_results))}"
            print(result)
            with open(Path(save_dir) / f"{lang}_correct.json", "w", encoding='utf-8') as f:
                json.dump(correct_results, f, ensure_ascii=False, indent=4)
            with open(Path(save_dir) / f"{lang}_wrong.json", "w", encoding='utf-8') as f:
                json.dump(wrong_results, f, ensure_ascii=False, indent=4)



def gsm8k_batch_gen(
    lang_, cur_gsm8k_batch, batch_llm, args
):
    lang = lang_ if lang_ != 'En_gsm8k' else 'English'
    prompt_no_input = (
      "Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
    )
    try:
        curs_gsm8k_questions = [v['question'] for v in cur_gsm8k_batch]
    except:
        curs_gsm8k_questions = [v['input_prompt'] for v in cur_gsm8k_batch]
    # prompt_no_input = PROMPT_DICTS['normal_prompt']
    input_str_list = [prompt_no_input.format(query=q) for q in curs_gsm8k_questions]
    output_str_list = batch_llm(input_str_list)
    return input_str_list, output_str_list


def get_batch_llama(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args):
    @torch.inference_mode()
    def batch_llama(input_strs):
        input_ids_w_attnmask = tokenizer(
            input_strs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        output_ids = model.generate(
            input_ids=input_ids_w_attnmask.input_ids,
            attention_mask=input_ids_w_attnmask.attention_mask,
            generation_config=GenerationConfig(
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.9,  # t=0.0 raise error if do_sample=True
            ),
        ).tolist()
        
        
        
        
        
        real_output_ids = [
            output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)
        ]
        output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
        return output_strs

    return batch_llama


def get_model(model_path: str, is_bf16: bool = False):
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    print(tokenizer.pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('new pad ', tokenizer.pad_token)
    print(tokenizer.bos_token)
    print(tokenizer.unk_token)
    print(tokenizer.eos_token)
    print(tokenizer.truncation_side)
    print(tokenizer.padding_side)

    if is_bf16:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
             device_map='auto'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
        ).cuda()
    model.eval()
    print(model.dtype)

    return model, tokenizer


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0


if __name__ == "__main__":
    import fire


    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batchsize",
        required=True
    )

    parser.add_argument(
        "--shot",
        type=int,
        help="how many examples in your prompts",
        default=4
    )
    parser.add_argument(
        "--shuffle",
        type= bool,
        help="whether to shuffle your choices",
        default = True
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="maximum output tokens",
        default = 1024
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed",
        default = 0
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="specific language to test",
        default = './dataset/train_set'
    )
    args = parser.parse_args()

    fire.Fire(main(args=args))
