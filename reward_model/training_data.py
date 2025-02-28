import re
import json
import os
os.chdir('/mnt/liao/planner')
from embedding_model import semb
import torch
from torch.utils.data import TensorDataset, random_split
from transformers import AutoTokenizer, AutoModel
from reward_model.agents_descriptions import code_agent_descriptions, math_agent_descriptions, search_agent_descriptions, commonsense_agent_descriptions
import pandas as pd

'''
This file aims to prepare the training dataset for the reward model.
We should create several plans, get responses, scores in advance.
The raw dataset should have the form like:
{
"task": ""
"agent": ""
"history": ""
("code": "" / "original_answer": "")
"response": ""
"original_query": ""
"score": ""
}
In this file we doesn't need to consider the relationship of the subtasks.
Each task corresponds to a agent and a score.
'''

# load the embedding model
tokenizer = AutoTokenizer.from_pretrained('/mnt/liao/planner/models/strans')
model = AutoModel.from_pretrained('/mnt/liao/planner/models/strans')

def reward_model_dataset(path = 'documents/responses_score_reason.txt', save_path_train = 'datasets/huskyqa-subtasks/train.pt', 
                         save_path_val = 'datasets/huskyqa-subtasks/val.pt', train_ratio = 0.9, model = model, tokenizer = tokenizer):

    # read in the responses and scores
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract contents in each ***[]***
    pattern = re.compile(r'\*\*\*\[(.*?)\]\*\*\*')
    matches = pattern.findall(content)

    # extract subtasks, descriptions, and scores
    subtasks = []
    descriptions = []
    scores = []
    for i in range(len(matches)):
        match = matches[i]
        # break
        match_json = json.loads('[' + match+ ']')
        for j in range(len(match_json)):
            for _ in range(10):
                scores.append(match_json[j]['score'])
                subtasks.append(match_json[j]['task'])
            agent_name = match_json[j]['agent']
            if agent_name == 'code_agent':
                for k in range(10):
                    descriptions.append(code_agent_descriptions[k])
            elif agent_name == 'math_agent':
                for k in range(10):
                    descriptions.append(math_agent_descriptions[k])
            elif agent_name == 'search_agent':
                for k in range(10):
                    descriptions.append(search_agent_descriptions[k])
            elif agent_name == 'commonsense_agent':
                for k in range(10):
                    descriptions.append(commonsense_agent_descriptions[k])

    # get the embeddings of the subtasks and descriptions
    subtasks_embd = semb(subtasks, model, tokenizer) 
    descriptions_embd = semb(descriptions, model, tokenizer)

    # 暂时先把subtasks的embeddings放在descriptions的前面，后续再看有没有影响
    x_tensor = torch.cat((subtasks_embd, descriptions_embd), 1)
    y_tensor = torch.tensor(scores).reshape(-1, 1)

    # 创建 TensorDataset
    dataset = TensorDataset(x_tensor, y_tensor)

    # 计算训练数据和验证数据的大小
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # 使用 random_split 函数将数据集分割
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # save the datasets
    torch.save(train_dataset, save_path_train)
    torch.save(val_dataset, save_path_val)

    return train_dataset, val_dataset
