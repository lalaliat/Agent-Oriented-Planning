import re
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from utils import semb
from utils import sim
import torch.nn.functional as F
import pandas as pd
'''
This file aims to create representations for each agent.
The raw dataset containing task, agent and score should be provided.
So after getting the responses and the scores of the subtasks, this representations can be formed.
'''

# load the embedding model
tokenizer = AutoTokenizer.from_pretrained('/mnt/liao/planner/models/strans')
model = AutoModel.from_pretrained('/mnt/liao/planner/models/strans')

def agent_rep(rep, threshold = 0.7, model = model, tokenizer = tokenizer):
    # 给定agents的一些代表作（score为10），筛除掉相似的sentences，留下彼此之间sim都小于threshold的代表作。
    rep_embd = semb(rep, model, tokenizer) 
    rep_sim = [rep[0]]
    rep_embd_sim = [rep_embd[0]]
    for i in range(1, len(rep)):
        flag = 0
        for j in range(len(rep_sim)):
            if F.cosine_similarity(rep_embd[i].unsqueeze(0), rep_embd_sim[j].unsqueeze(0)) > threshold:
                flag = 1
                break
        if flag == 0:
            rep_sim.append(rep[i])
            rep_embd_sim.append(rep_embd[i])
    return rep_sim, rep_embd_sim

path = '/mnt/liao/planner/documents/responses.txt'

# keep the representatons with low similarity
def representations(path, threshold = 0.9, score = 10):
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = re.compile(r'\*\*\*\[(.*?)\]\*\*\*')
    matches = pattern.findall(content)

    # collect all the subtasks with score equals to 10.
    rep = []
    for i in range(len(matches)):
        match = matches[i]
        # break
        match_json = json.loads('[' + match + ']')
        for j in range(len(match_json)):
            if match_json[j]['score']>=score:
                rep.append(match_json[j])

    # See how many subtasks with score 10 are there.
    agent_name = [rep[i]['agent'] for i in range(len(rep))]
    frequency = Counter(agent_name)
    print(f"Frequency: {frequency}")

    code_rep = [rep[i]['task'] for i in range(len(rep)) if rep[i]['agent']=='code_agent']
    math_rep = [rep[i]['task'] for i in range(len(rep)) if rep[i]['agent']=='math_agent']
    search_rep = [rep[i]['task'] for i in range(len(rep)) if rep[i]['agent']=='search_agent']
    commonsense_rep = [rep[i]['task'] for i in range(len(rep)) if rep[i]['agent']=='commonsense_agent']

    code_rep, code_rep_embd = agent_rep(code_rep, threshold)
    math_rep, math_rep_embd = agent_rep(math_rep, threshold)
    search_rep, search_rep_embd = agent_rep(search_rep, threshold)
    commonsense_rep, commonsense_rep_embd = agent_rep(commonsense_rep, threshold)

    return code_rep, code_rep_embd, math_rep, math_rep_embd, search_rep, search_rep_embd, commonsense_rep, commonsense_rep_embd

def representations_excel(path, score, threshold = 0.8):
    df = pd.read_excel(path)
    code_rep = list(set(df[(df['score'] == score) & (df['agent'] == 'code_agent')]['task']))
    math_rep = list(set(df[(df['score'] == score) & (df['agent'] == 'math_agent')]['task']))
    search_rep = list(set(df[(df['score'] == score) & (df['agent'] == 'search_agent')]['task']))
    commonsense_rep = list(set(df[(df['score'] == score) & (df['agent'] == 'commonsense_agent')]['task']))

    code_rep, code_rep_embd = agent_rep(code_rep, threshold)
    math_rep, math_rep_embd = agent_rep(math_rep, threshold)
    search_rep, search_rep_embd = agent_rep(search_rep, threshold)
    commonsense_rep, commonsense_rep_embd = agent_rep(commonsense_rep, threshold)

    return code_rep, code_rep_embd, math_rep, math_rep_embd, search_rep, search_rep_embd, commonsense_rep, commonsense_rep_embd
