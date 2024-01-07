import os
import openai
openai.api_type = "azure"
openai.api_base = "https://new-llm.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "082b19bc29364b1bb39d2d9fb9b757d4"

def GPT4_Inference(prompt):
  prompt = [{"role": "system", "content": "You are an assistant that fixs some graph reasoning tasks."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": prompt}]
  response = openai.ChatCompletion.create(
    engine="gpt-35-turbo",
    messages = prompt,
    temperature=0.1,
    max_tokens=1000,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)
  
  return response["choices"][0]["message"]["content"]

prompt = "Determine if there is a path between two nodes in an undirected graph. In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. The nodes are numbered from 0 to 37, and the edges are: (0, 11) (0, 30) (0, 22) (0, 13) (0, 21) (1, 33) (1, 19) (1, 10) (1, 9) (1, 27) (1, 18) (1, 34) (2, 5) (2, 15) (2, 36) (2, 8) (3, 10) (4, 27) (5, 12) (5, 17) (5, 31) (6, 7) (6, 26) (6, 12) (7, 31) (7, 36) (7, 32) (7, 29) (8, 16) (8, 10) (9, 23) (9, 25) (10, 18) (10, 15) (11, 12) (11, 18) (11, 14) (11, 19) (11, 31) (11, 20) (12, 24) (13, 19) (13, 36) (14, 35) (14, 22) (16, 33) (17, 24) (17, 35) (18, 26) (19, 20) (20, 30) (20, 23) (21, 35) (21, 29) (21, 31) (23, 28) (25, 26) (26, 36) (28, 34) (30, 34) (33, 37) (35, 37). Q: Is there a path between node 30 and node 11? Output Yes or No. Begin with '###' to give your final conclusion. Let's think step by step."
print(GPT4_Inference(prompt))