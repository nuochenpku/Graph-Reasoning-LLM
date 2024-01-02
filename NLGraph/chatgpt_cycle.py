import json
import openai
import time
# openai.api_type = "azure"
from pathlib import Path
from tqdm import tqdm
openai.api_type = "azure"
openai.api_base = "https://new-llm.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "082b19bc29364b1bb39d2d9fb9b757d4"

def chatGPT_test(text, content=None):

    prompt = [{"role": "system", "content": 'You are a helpful assistant that are good at providing detailed explanations to solve graph reasoning tasks.'},
        {"role": "user", "content": "{}".format(text)}]
    i = 0
    responses = []
    while True:
        try:
            if i>=3:
                break
            completion = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages = prompt,
                temperature=0.7,
                max_tokens=600,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0)
            break
        except Exception as e:
            i+=1
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            # import time
            time.sleep(30)
    time.sleep(10.0)
    return completion["choices"][0]["message"]["content"]
with open("/cpfs/user/chennuo/CN/NLGraph/NLGraph/cycle/prompt/" + "easy-CoT-prompt.txt", "r") as f:
    exemplar = f.read()

with open('/cpfs/user/chennuo/CN/NLGraph/NLGraph/cycle/train.json') as f:
    datas = json.load(f)

save_path ='/cpfs/user/chennuo/CN/NLGraph/NLGraph/cycle/chatgpt_tmp07_2'
Path(save_path).mkdir(parents=True, exist_ok=True)  

gen_datas_jsonl = Path(save_path) / '_generate_train.json'

 
start_index = (
        len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
    )
print(f"start_index: {start_index}")
    
for key, value in tqdm(datas.items()):
    
    if int(key) <= start_index:
        continue
    
    query = value['question']
    instruct = "Determine whether or not there is a cycle in an undirected graph. Begin with '###' to give your final conclusion.\n" + query.split('\n')[0] 
    query = query.replace(query.split('\n')[0], instruct)
    # query += "\nLet's think step by step:"
    inputs = exemplar + '\n\n\n' + query
    
    # inputs += "\n You should keep in mind that you should write a summarized sentence at last and begin with '###' to give your final conclusion."
    response = chatGPT_test(inputs)
    # print(inputs)
    # print(response)
    # exit()
    
    try:
        response = chatGPT_test(inputs)
        new_data = dict()
        value['chatgpt_response'] = response
        new_data[key] = value
        

        
        with open(gen_datas_jsonl, "a") as w:
            w.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            
        if 'yes' in value['answer'] and 'yes' in response.split('###')[-1].lower():
            with open(Path(save_path)/'_generate_train_correct.json', "a") as w:
                w.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                
        elif 'no' in value['answer'] and 'no' in response.split('###')[-1].lower():
            with open(Path(save_path)/'_generate_train_correct.json', "a") as w:
                w.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        else:
            with open(Path(save_path)/'_generate_train_wrong.json', "a") as w:
                w.write(json.dumps(new_data, ensure_ascii=False) + '\n')
    except:
        continue