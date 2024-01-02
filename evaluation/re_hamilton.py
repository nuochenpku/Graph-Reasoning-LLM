import openai
import os
from tqdm import tqdm
import networkx as nx
import numpy as np
import argparse
import time
from datetime import datetime, timedelta, timezone
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import json
model_list = ["text-davinci-003","code-davinci-002","gpt-3.5-turbo","gpt-4"]
parser = argparse.ArgumentParser(description="hamilton")
parser.add_argument('--model', type=str, default="text-davinci-003", help='name of LM (default: text-davinci-003)')
parser.add_argument('--mode', type=str, default="hard", help='mode (default: easy)')
parser.add_argument('--prompt', type=str, default="none", help='prompting techniques (default: none)')
parser.add_argument('--T', type=int, default=0, help='temprature (default: 0)')
parser.add_argument('--token', type=int, default=400, help='max token (default: 400)')
parser.add_argument('--SC', type=int, default=0, help='self-consistency (default: 0)')
parser.add_argument('--SC_num', type=int, default=5, help='number of cases for self-consistency (default: 5)')
args = parser.parse_args()
prompt_list = ["CoT", "none", "0-CoT", "LTM", "PROGRAM","k-shot","Algorithm","Instruct"]
i = 2
num_shot = ["12-shot"]
while i <= 16:
    num_shot.append(str(i)+"-shot")
    i *= 2
prompt_list = prompt_list + num_shot
assert args.prompt in prompt_list

def translate(G, args):
    edge = list(G.edges())
    n, m = G.number_of_nodes(), G.number_of_edges()
    Q = ''
    if args.prompt in ["CoT", "k-shot","Algorithm","Instruct"] + num_shot:
        with open("NLGraph/hamilton/prompt/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    Q = Q + "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.\nThe nodes are numbered from 0 to " + str(n-1)+", and the edges are:"
    #character = [chr(65+i) for i in range(26)] + [chr(65+i)+chr(65+i) for i in range(26)]
    for i in range(len(edge)):
        Q = Q + ' ('+str(edge[i][0])+','+str(edge[i][1])+')'
    Q = Q + "\n"
    if args.prompt == "Instruct":
        Q = Q + "Let's construct a graph with the nodes and edges first.\n"
    Q = Q + "Q: Is there a path in this graph that visits every node exactly once? If yes, give the path. Note that in a path, adjacent nodes must be connected with edges.\nA:"
    if args.prompt == "0-CoT":
            Q = Q + " Let's think step by step:"
    elif  args.prompt == "LTM":
            Q = Q + " Let's break down this problem:" 
    elif args.prompt == "PROGRAM":
            Q = Q + " Let's solve the problem by a Python program:"
    return Q

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1000))
def predict(Q):
    input = Q
    temperature = 0
    if args.SC == 1:
        temperature = 0.7
    if 'gpt' in args.model:
        Answer_list = []
        for text in input:
            response = openai.ChatCompletion.create(
            model=args.model,
            messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=args.token,
            )
            Answer_list.append(response["choices"][0]["message"]["content"])
        return Answer_list
    response = openai.Completion.create(
    model=args.model,
    prompt=input,
    temperature=temperature,
    max_tokens=args.token,
    )
    Answer_list = []
    for i in range(len(input)):
        Answer_list.append(response["choices"][i]["text"])
    return Answer_list

def log(Q, res, answer, args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    time = bj_dt.now().strftime("%Y%m%d---%H-%M")
    newpath = 'log/hamilton/'+args.model+'-'+ args.mode+'-'+time+ '-' + args.prompt
    if args.SC == 1:
        newpath = newpath + "+SC"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    newpath = newpath + "/"
    np.save(newpath+"res.npy", res)
    np.save(newpath+"answer.npy", answer)
    with open(newpath+"prompt.txt","w") as f:
        f.write(Q)
        f.write("\n")
        f.write("Acc: " + str((res==1).sum())+'/'+str(len(res)) + '\n')
        f.write("Acc2: " + str((res>=0).sum())+'/'+str(len(res)) + '\n')
        f.write("\n")
        print(args, file=f)

def check(solution, G):
    n = G.number_of_nodes()
    if len(solution) != n:
        return 0
    for i in range(len(solution)-1):
        if not G.has_edge(solution[i], solution[i+1]):
            #print(solution[i], solution[i+1])
            return 0
    for i in range(len(solution)-1):
        for j in range(i+1, len(solution)-1):
            if solution[i] == solution[j]:
                return 0
    return 1
def process_ans(ans, pos, G):
    num, flag = 0, 0
    solution = []
    n = G.number_of_nodes()
    for i in range(pos, len(ans)):
        if ans[i] >= '0' and ans[i] <='9':
            num = num*10 + int(ans[i])
            flag = 1
        else:
            if flag == 1:
                solution.append(num)
                if len(solution) == n:
                    break
                flag = 0
            num = 0
    return solution

def evaluate(ans, G):
    pos = ans.find("the path can be") # for codex
    pos2 = ans.find("no ")
    if pos2 == -1:
        pos2 = 10000000
    if pos2 < pos:
        return -1
    if pos == -1:
        return -1
    solution = process_ans(ans, pos, G)
    flag  = check(solution, G)
    return flag

def main():
    # if 'OPENAI_API_KEY' in os.environ:
    #     openai.api_key = os.environ['OPENAI_KEY']
    # else:
    #     raise Exception("Missing openai key!")
    # if 'OPENAI_ORGANIZATION' in os.environ:
    #     openai.organization = os.environ['OPENAI_ORGANIZATION']
    res, answer = [], []
    
    if args.mode== "easy":
            g_num = 150*2
    else:
            g_num = 600

    batch_num = 20
    #g_num = 10
    G_list, Q_list = [], []
    edges = []
    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        # G_list, Q_list = [], []
        for j in range(i*batch_num, min(g_num, (i+1)*batch_num)):
            with open("/cpfs/user/chennuo/CN/NLGraph/NLGraph/hamilton/graph/"+args.mode+"/full/graph"+str(j)+".txt","r") as f:
                n, m = [int(x) for x in next(f).split()]
                edge = []
                for line in f: # read rest of lines
                    edge.append([int(x) for x in line.split()])
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for k in range(m):
                    G.add_edge(edge[k][0], edge[k][1])
                Q = translate(G, args)
                Q_list.append(Q)
                G_list.append(G)
                edges.append(edge)

    print(len(Q_list))
    print(len(G_list))
    with open('/cpfs/user/chennuo/CN/NLGraph/NLGraph/hamilton/hamilton_full.json', 'a') as w:
        for Q, G, E in zip(Q_list, G_list, edges):
            # for Q, G in zip(Qs, Gs):
            new_data = dict()
            new_data['question'] = Q
            new_data['graph'] = str(G)
            new_data['edge'] = E
            new_data["difficulty"]= args.mode
            w.write(json.dumps(new_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()