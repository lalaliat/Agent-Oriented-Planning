import requests
import os
os.chdir('/mnt/liao/planner')
from utils import score
from reward_model.agents_descriptions import code_agent_descriptions, math_agent_descriptions, search_agent_descriptions, commonsense_agent_descriptions
from prompt import redescribe_subtask_prompt, planner_prompt, plan_in_detail_prompt
from utils import semb, is_valid_json
from reward_model.representations import representations
from MLP import SimilarityMLP
from transformers import AutoTokenizer, AutoModel
import torch
import json
import torch.nn.functional as F
import re


'''
This file aims to build the planner given the prompt.
'''

tokenizer = AutoTokenizer.from_pretrained('/mnt/liao/planner/models/strans')
model = AutoModel.from_pretrained('/mnt/liao/planner/models/strans')

class planner_gpt:
    def __init__(self):
        with open("keys/gptapi_key.json", "r") as f:
            data = json.load(f)
        self.Authorization = data["Authorization"]
        self.url = data["url"]
    def plan(self, query, planner_prompt_predefined = planner_prompt, model = "gpt-4o"):
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.Authorization
        }
        data = {
            "model": model,
            "messages": [
                {"role":"system", "content": planner_prompt_predefined},
                {"role": "user", "content": query}],
            "n": 1,
            "temperature": 0.0
        }
        response = requests.post(self.url, json=data, headers=headers)
        return response.json()
    
def claude(prompt, model="claude-2.1"):
    with open("keys/claudeapi_key.json", "r") as f:
        data = json.load(f)
    url = data["url"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": data["Authorization"]
    }
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens_to_sample": 2048,
        "stream": False
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()

    
def agent(prompt, model="gpt-4o"):
    with open("keys/gptapi_key.json", "r") as f:
        data = json.load(f)
    url = data["url"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": data["Authorization"]
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}],
        "n": 1,
        "temperature": 0.0
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()
    
planner = planner_gpt()


def plan_reward(plan, descriptions, model, tokenizer, MLP):
    plan_score = []
    for i in range(len(plan)):
        subtask_score = {}
        for agent_name in descriptions.keys():
            subtask_score[agent_name] = round(score(plan[i]['task'], descriptions[agent_name], model, tokenizer, MLP).item(),4)
        plan_score.append(subtask_score)
    return plan_score


# Modified for rebuttal
def plan_modify(plan, descriptions, MLP, agent_rep, agent_rep_embd, query, threshold_up = 0.625, 
                threshold_down = 0.125, threshold_similar = 0.7, model = model, tokenizer = tokenizer):
    '''
    plan is a list whose values are json varaible
    '''
    original_subtasks = ''
    for temp_i in range(len(plan)):
        original_subtasks += str(temp_i+1) + ': ' + plan[temp_i]['task'] + '\n'
    agent_name = list(descriptions.keys())
    plan_score = plan_reward(plan, descriptions, model, tokenizer, MLP)
    for i in range(len(plan)):
        print(plan[i]['task'])
        name1 = plan[i]['name_1']
        name2 = plan[i]['name_2']
        name3 = [name for name in agent_name if name != name1 and name != name2][0]
        name4 = [name for name in agent_name if name != name1 and name != name2][1]
        score1 = plan_score[i][name1]
        score2 = plan_score[i][name2]
        score3 = plan_score[i][name3]
        score4 = plan_score[i][name4]
        scores = [score1, score2, score3, score4]
        names = [name1, name2, name3, name4]

        if scores[0] >= threshold_up:
            plan[i]['agent'] = names[0]
        elif max(scores[1:]) >= threshold_up:
            plan[i]['agent'] = names[1 + scores[1:].index(max(scores[1:]))]
        elif max(scores)<threshold_down:
            # TODO: skip
            print('This subtask cannot be accomplished by any agent.')
            plan[i] = plan[i]
        else:
            task_embd = semb(plan[i]['task'], model, tokenizer)
            cos_name1 = F.cosine_similarity(task_embd, torch.stack(agent_rep_embd[name1]))
            cos_name2 = F.cosine_similarity(task_embd, torch.stack(agent_rep_embd[name2]))
            cos_name3 = F.cosine_similarity(task_embd, torch.stack(agent_rep_embd[name3]))
            cos_name4 = F.cosine_similarity(task_embd, torch.stack(agent_rep_embd[name4]))
            # print((torch.max(cos_name1), torch.max(cos_name2), torch.max(cos_name3), torch.max(cos_name4)))
            if any(x >= threshold_similar for x in (torch.max(cos_name1), torch.max(cos_name2), torch.max(cos_name3), torch.max(cos_name4))):
                if torch.max(cos_name1) >= threshold_similar:
                    name = name1
                    cos_name = cos_name1
                elif torch.max(cos_name2) >= torch.max(cos_name3) and torch.max(cos_name2) >= torch.max(cos_name4):
                    name = name2
                    cos_name = cos_name2
                elif torch.max(cos_name3) >= torch.max(cos_name4) or torch.max(cos_name3) >= torch.max(cos_name2):
                    name = name3
                    cos_name = cos_name3
                elif torch.max(cos_name4) >= torch.max(cos_name3) and torch.max(cos_name4) >= torch.max(cos_name2):
                    name = name4
                    cos_name = cos_name4
                similar_task = agent_rep[name][torch.where(cos_name == torch.max(cos_name))[0].item()]
                # redescribe
                prompt = redescribe_subtask_prompt % (similar_task, plan[i]['task'])
                rewrite_task = None
                for temp in range(5):
                    res = agent(prompt)
                    if isinstance(res.get('data'), dict):
                        rewrite_task = re.search(r'\*{3}(.*?)\*{3}', res['data']['response']['choices'][0]['message']['content'])
                        if rewrite_task:
                                rewrite_task = rewrite_task.group(1)
                                break
                        else:
                            rewrite_task = res['data']['response']['choices'][0]['message']['content']
                          
                plan[i]['agent'] = name
                if not rewrite_task:
                    rewrite_task = res['data']['response']['choices'][0]['message']['content']
                plan[i]['task'] = rewrite_task
            else:
                # get the plan_in_detail subtasks
                for temp in range(5):
                    res = agent(plan_in_detail_prompt % (query, original_subtasks, plan[i]['task']))
                    if isinstance(res.get('data'), dict):
                        # replace it with the original subtask, it becomes a list
                        # then if the length of subtask(s) belong to one id is larger than one, we need to summarize the responses of these subtasks
                        subtask_plan_in_detail = res['data']['response']['choices'][0]['message']['content']
                        start_index = subtask_plan_in_detail.find("```json") + len("```json")  # 找到第一个三引号，并且加3以排除三引号本身
                        end_index = subtask_plan_in_detail.rfind("```")
                        subtask_plan_in_detail = subtask_plan_in_detail[start_index:end_index].strip()
                        if is_valid_json(subtask_plan_in_detail):
                            plan_in_detail = json.loads(subtask_plan_in_detail)
                            plan[i] = plan_in_detail
                            print(plan[i])
                            # just choose the agent with higher score
                            plan_in_detail_score = plan_reward(plan[i], descriptions, model, tokenizer, MLP)
                            for j in range(len(plan[i])):
                                plan_in_detail_name1 = plan[i][j]['name_1']
                                plan_in_detail_name2 = plan[i][j]['name_2']
                                plan_in_detail_score1 = plan_in_detail_score[j][plan_in_detail_name1]
                                plan_in_detail_score2 = plan_in_detail_score[j][plan_in_detail_name2]
                                plan[i][j]['agent'] = plan_in_detail_name1 if plan_in_detail_score1 >= plan_in_detail_score2 else plan_in_detail_name2
                            break
    return plan
