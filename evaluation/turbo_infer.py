import os
import openai
import json
openai.api_type = "azure"
openai.api_base = "https://new-llm.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "082b19bc29364b1bb39d2d9fb9b757d4"

def GPT4_Inference(prompt, temp):
  prompt = [{"role": "system", "content": "You are an assistant that fixs some graph reasoning tasks."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
    engine="gpt-35-turbo",
    messages = prompt,
    temperature=temp,
    max_tokens=1000,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)
  
  return response["choices"][0]["message"]["content"]

with open("../datasets/train_set/topology_train.json", "r") as f:
  index = 0
  for line in f.readlines():
    line = json.loads(line)
    prompt = line["input_prompt"]
    for temp in (0.1, 0.3, 0.5, 0.7, 0.9):
      print(line["answer"])
      print(prompt)
      print(GPT4_Inference(prompt, temp))
    index += 1
    if index == 10:
      break
