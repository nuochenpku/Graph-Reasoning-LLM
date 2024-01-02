import re
import openai
import json
from tqdm import tqdm
import time

import re
def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0


openai.api_type = "azure"
openai.api_base = "https://llm.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
openai.api_key = '5d93a0feea2a46ddac086d7d4d37d6ea'
def chatGPT_test(text, lang, content):
    # prompt = [{"role": "system", "content": "Assuming you are a agile and professional assistant, when given a query, you can generate corresponding response."},
    #     {"role": "user", "content": "Hello!"},
    #     {"role": "assistant", "content": "Hello!"},
    #     {"role": "user", "content": "{}".format(text)}]
  
    prompt = [{"role": "system", "content": f"Answer the following math probelm step by step in {lang}.\n{content}"},
        {"role": "user", "content": "{}".format(text)}]

    
    # prompt = [{"role": "system", "content": "You will get a task description and you need to finish the task based on the content of the task description."},
    #     {"role": "user", "content": "{}".format(text)}]
    #GPT3 text-davinci-003
    #chatGPT gpt-3.5-turbo
    i = 0
    while True:
        try:
            if i>=3:
                break
            # completion = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            # messages=prompt,
            # max_tokens=300,
            # top_p=1,
            # temperature=0.0
            # )
            
            completion = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages = prompt,
                temperature=0.0,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            break
        except Exception as e:
            i+=1
            tqdm.write(str(e))
            tqdm.write("Retrying...")
            # import time
            time.sleep(30)
    # for item in prompt:
    #     print(item["role"], ":", item["content"])
    # print(completion["choices"][0]["message"]["content"])
    time.sleep(2.0)
    return completion["choices"][0]["message"]["content"]

examplars = """

Q: julia played tag with 18 kids on monday . she played tag with 10 kids on tuesday . how many more kids did she play with on monday than on tuesday ?
A: Let's think step by step. julia playsed tag with 18 kids on monday and 10 kids tuesday, separately. So the amount of kids that she played with on monday than on tuesday is 18-10=8. So the answer is 8.

Q: Jack had 9 action figures and 10 books on a shelf in his room . later he added 7 more action figures to the shelf . how many more action figures than books were on his shelf ?
A: Let's think step by step. The amount of action figures that Jack had is 9+7=16. And Jack had 10 books. So the amount of action figures than books on his shelf is 16-10=6. So the answer is 7.
\n
"""

results = dict()
# langs = ['English','Bengali', 'Chinese', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'Swahili', 'Thai']
tasks = ['connectivity', 'cycle','flow', 'GNN', 'hamilton', 'matching','shortest_path', 'topology']
with open('gpt4_nlg_sample3.json',  'w', encoding='utf-8') as writer:
    for task in tasks:
        rights = 0
        # data_path = f"/cpfs/user/chennuo/CN/gsm8k-ScRel/data/mgsm/mgsm_{lang}.json"
        # acc, folder = main(args, datas[:16])
        
        
        with open(f"/cpfs/user/chennuo/CN/NLGraph/NLGraph/{task}/prompt/" + "CoT-prompt.txt", "r") as f:
            exemplar = f.read()


        for key, data in tqdm(datas.items()):
            response = data['answer']
            
            query = data['question']

            inputs = exemplar + '\n\n\n' + query + " \nBegin with '###' to give your final conclusion."
                    
            
            new_data = {}
            new_data['prompt'] = inputs
            new_data['max_tokens'] = 1050
            new_data['temperature'] = 0.7
            new_data['n'] = int(1)
            new_data['top_p']= int(1)
            new_data['frequency_penalty']= int(0)
            new_data['presence_penalty']= int(0)
            new_data['stop']= None
            new_data['response'] = response
            new_data['query'] = query
            new_data['task'] = task
            new_data['id'] = key
            
            writer.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            writer.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            writer.write(json.dumps(new_data, ensure_ascii=False) + '\n')
