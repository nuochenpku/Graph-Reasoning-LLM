# Graph-Reasoning-LLM
this is project for training explicit graph reasoning large language models.




## Usage & Download

Our checkpoints and dataset are avaliable at HuggingFace. You can directly download them according to the following links:

### 🤗Models🤗


| GraphWiz   	| Mixed-Task Training                                       	| DPO                                                       	|
|-----	|---------------------------------------------------------------	|---------------------------------------------------------------------------	|
| 🤗**7B-LLaMA 2**  	| 🐙 [GraphWiz-7B](https://huggingface.co/GraphWiz/Parallel_7B)   	| 🐙 [GraphWiz-7B](https://huggingface.co/GraphWiz/Cross_7B)  	|
| 🤗**13B-LLaMA 2** 	| 🐙 [GraphWiz-13B](https://huggingface.co/GraphWiz/Parallel_13B/tree/main) 	| 🐙 [GraphWiz-13B](https://huggingface.co/GraphWiz/Cross_13B)	|
| 🤗**7B-Mistral** 	| [**GrpahWiz-7B**](https://huggingface.co/GraphWiz/Mistral-7B), [**GrpahWiz-7B-RFT**](https://huggingface.co/GraphWiz/Mistral-7B-RFT)   	| 🐙 [GraphWiz-DPO]() 	|



## Training GraphWiz 

### Requirements

```
pip -r install requirements.txt
```

### Phase1: Mixed-Task Training 



### Requirements

```
pip -r install requirements.txt
```









## GraphInstruct  Construction
If you want to construct additional graph problem data for training your own models for graph problem reasoning. Please refer to the following:

### Generate Graph Problems

#### step1: Generate all train set

```
cd scripts
bash generate_all_train_datasets.sh
```

#### Generate all test set

```
cd scripts
bash generate_all_test_datasets.sh
```

### Step2: Data Augmentation with Rejection Sampling
Here, we introduce how to select diverse paths for dpo training data:

#### Step2.1: Inference SFT model multiple times
 Suppose we already have the sft model. You can directly use our models at HuggingFace: [**GraphWiz**](https://huggingface.co/GraphWiz)
```
cd evaluation
bash rft.sh
```
The default inference times 'seed' is set to 20.

#### Step2.2: Select the diverse paths 

Then we filter out the diverse reasoning paths:

```
cd find_paths
python3 select_path_dpo.py

python3 find_path.py
```

Please note that you should changle the data paths according to your local enviroment.

At last, you can obtain the json file like:

```
 "0": {
        "neg_response": [
            "To find the maximum flow from node 4 to node 8, we need to consider all the possible paths from node 4 to node 7, and then from node 7 to node 8. \nOne such path is 4->5->7->8, which can carry a flow of 6 units, as limited by the edge (4->5).\nAnother path is 4->6->7->8, which can carry a flow of 7 units, as limited by the edge (4->6).\nHowever, not all of this flow can reach node 8, because the edge (7->8) has a capacity of 3 units, and the edge (6->8) has a capacity of 10 units. Therefore, the bottleneck of this path is the edge (7->8), which reduces the flow to 3 units.\nTherefore, by adding the flows of the two paths, we get the maximum flow from node 4 to node 8 as 6 + 7 = 13 units.\n### The maximum flow from node 4 to node 8 is 13 units.",
            "The maximum flow from node 4 to node 8 is equal to the sum of the flows on the edges leaving node 4, which is 7 + 6 = 13 units. ### 13.",
            "To find the maximum flow from node 4 to node 8, we need to consider all possible paths from node 4 to node 8 and their capacities. One such path is 4->5->7->8, which has a capacity of 6 units, as limited by the edge (4->5). Another path is 4->6->8, which has a capacity of 7 units, as limited by the edge (4->6). However, note that the edge (6->8) has a smaller capacity of 10 units, so it cannot carry all the flow from node 4 to node 8 via this path. Therefore, we need to hybridize the two paths and find a way to allocate the flow between them without compromising the edge capacities. One possible allocation is to send 6 units from node 4 to node 5, 3 units from node 5 to node 7, and 7 units from node 4 to node 6, which gives a total flow of 6 + 3 + 7 = 16 units. However, this allocation is not optimal, as it wastefully converts 6 units of flow from node 4 to node 5, which is then unnecessarily routed to node 7. A more optimal allocation is to send 3 units from node 4 to node 5, 6 units from node 5 to node 7, and 7 units from node 4 to node 6, which gives a total flow of 3 + 6 + 7 = 16 units. This allocation avoids the conversion of any flow from node 4 to node 5, which is then routed to node 7. Hence, the maximum flow from node 4 to node 8 is 16 units, which is the sum of the flows along the two paths.### 16.",
            "### The maximum flow from node 4 to node 8 is 13 units."
        ],
        "pos_response": [
            "To find the maximum flow from node 4 to node 8, we need to consider all possible paths from node 4 to node 8 and their capacities. \nOne such path is through node 5, where node 4 can send up to 6 units to node 5, but node 5 can only forward up to 10 units to node 7, and node 7 can only send up to 3 units to node 8. This path can carry a maximum of 3 units of flow. \nAnother path is through node 6, where node 4 can send up to 7 units to node 6, and node 6 can forward up to 10 units to node 8. This path can carry a maximum of 7 units of flow. \nHowever, note that the capacity of the edge from node 6 to node 8 is 10 units, so it cannot carry more flow than that regardless of the incoming flow from node 4. \nTherefore, by adding the flows along the two paths, we find that the maximum flow from node 4 to node 8 is 10 units. ### 10.",
            "The relevant edges for the flow from node 4 to node 8 are (4->5,6), (4->6,7), (5->7,10), and (6->8,10).\nOne possible path is from node 4 to node 5 and then from node 5 to node 7, and then from node 7 to node 8. This path can carry a flow of 10 units, as limited by the edge (5->7).\nAnother possible path is from node 4 to node 6 and then from node 6 to node 8. This path can carry a flow of 7 units, as limited by the edge (6->8).\nHowever, these two paths share the edge (6->8), which has a capacity of 10 units. Therefore, the total flow that can be sent from node 4 to node 8 cannot exceed 10 units, regardless of the mixture of paths.\nHence, the maximum flow from node 4 to node 8 is 10 units. ### 10."
        ],
        "task": "flow",
        "CoT_response": "To find the maximum flow from node 4 to node 8, we need to consider all possible paths from node 4 to node 8 and their capacities. \nOne such path is through node 5, where node 4 can send up to 6 units to node 5, but node 5 can only forward up to 10 units to node 7, and node 7 can only send up to 3 units to node 8. This path can carry a maximum of 3 units of flow. \nAnother path is through node 6, where node 4 can send up to 7 units to node 6, and node 6 can forward up to 10 units to node 8. This path can carry a maximum of 7 units of flow. \nHowever, note that the capacity of the edge from node 6 to node 8 is 10 units, so it cannot carry more flow than that regardless of the incoming flow from node 4. \nTherefore, by adding the flows along the two paths, we find that the maximum flow from node 4 to node 8 is 10 units. ### 10.",
        "pos_sort": {
            "edit": [
                0,
                1
            ],
            "jaccard": [
                0,
                1
            ],
            "tfidf": [
                0,
                1
            ],
            "cosine": [
                0,
                1
            ]
        },
        "neg_sort": {
            "edit": [
                0,
                1,
                3,
                2
            ],
            "jaccard": [
                0,
                2,
                1,
                3
            ],
            "tfidf": [
                2,
                0,
                1,
                3
            ],
            "cosine": [
                0,
                2,
                1,
                3
            ]
        },
        "pos_rft_paths_5": [
            "The relevant edges for the flow from node 4 to node 8 are (4->5,6), (4->6,7), (5->7,10), and (6->8,10).\nOne possible path is from node 4 to node 5 and then from node 5 to node 7, and then from node 7 to node 8. This path can carry a flow of 10 units, as limited by the edge (5->7).\nAnother possible path is from node 4 to node 6 and then from node 6 to node 8. This path can carry a flow of 7 units, as limited by the edge (6->8).\nHowever, these two paths share the edge (6->8), which has a capacity of 10 units. Therefore, the total flow that can be sent from node 4 to node 8 cannot exceed 10 units, regardless of the mixture of paths.\nHence, the maximum flow from node 4 to node 8 is 10 units. ### 10."
        ],
        "neg_rft_paths_5": [
            "To find the maximum flow from node 4 to node 8, we need to consider all the possible paths from node 4 to node 7, and then from node 7 to node 8. \nOne such path is 4->5->7->8, which can carry a flow of 6 units, as limited by the edge (4->5).\nAnother path is 4->6->7->8, which can carry a flow of 7 units, as limited by the edge (4->6).\nHowever, not all of this flow can reach node 8, because the edge (7->8) has a capacity of 3 units, and the edge (6->8) has a capacity of 10 units. Therefore, the bottleneck of this path is the edge (7->8), which reduces the flow to 3 units.\nTherefore, by adding the flows of the two paths, we get the maximum flow from node 4 to node 8 as 6 + 7 = 13 units.\n### The maximum flow from node 4 to node 8 is 13 units.",
            "To find the maximum flow from node 4 to node 8, we need to consider all possible paths from node 4 to node 8 and their capacities. One such path is 4->5->7->8, which has a capacity of 6 units, as limited by the edge (4->5). Another path is 4->6->8, which has a capacity of 7 units, as limited by the edge (4->6). However, note that the edge (6->8) has a smaller capacity of 10 units, so it cannot carry all the flow from node 4 to node 8 via this path. Therefore, we need to hybridize the two paths and find a way to allocate the flow between them without compromising the edge capacities. One possible allocation is to send 6 units from node 4 to node 5, 3 units from node 5 to node 7, and 7 units from node 4 to node 6, which gives a total flow of 6 + 3 + 7 = 16 units. However, this allocation is not optimal, as it wastefully converts 6 units of flow from node 4 to node 5, which is then unnecessarily routed to node 7. A more optimal allocation is to send 3 units from node 4 to node 5, 6 units from node 5 to node 7, and 7 units from node 4 to node 6, which gives a total flow of 3 + 6 + 7 = 16 units. This allocation avoids the conversion of any flow from node 4 to node 5, which is then routed to node 7. Hence, the maximum flow from node 4 to node 8 is 16 units, which is the sum of the flows along the two paths.### 16."
        ],
        "query": "Find the maximum flow between two nodes in a directed graph. In a directed graph, (i->j,k) means that node i and node j are connected with an directed edge from node i to node j with weight k. Given a graph and a pair of nodes, you need to output the maximum flow between the two nodes. Q: The nodes are numbered from 0 to 8, and the edges are: (0->7,2) (0->3,9) (1->3,2) (2->3,2) (2->5,4) (4->5,6) (4->6,7) (5->7,10) (6->8,10) (6->7,9) (7->8,3). What is the maximum flow from node 4 to node 8?"
    }

```
- "**pos_rft_paths_5**" refers to the diverse ***Correct*** reasoning paths (<=5);
- "**neg_rft_paths_5**" refers to the diverse ***InCorrect*** reasoning paths (<=5).






## Trl Training

change Llama_path & evaluation file path.

```
# dpo
python training/dpo.py 
# sft
python training/sft.py
```
