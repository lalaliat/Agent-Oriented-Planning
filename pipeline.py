import os
os.chdir('/mnt/liao/planner')
import datasets
from planner import planner_gpt, plan_modify
from utils import is_valid_json, level_score
import json
from MLP import SimilarityMLP
import torch
from transformers import AutoTokenizer, AutoModel
from reward_model.agents_descriptions import code_agent_descriptions, math_agent_descriptions, search_agent_descriptions, commonsense_agent_descriptions
from reward_model.representations import representations
from utils import score, query_desc_embd, semb, extract_content_between_markers, find_max_position
import re
from get_response import get_response, plan_execution
from prompt import evaluate_prompt
from planner import agent
import time
import random
from prompt import scorer_prompt
import pandas as pd
import openpyxl
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch.optim as optim

# region prepare the information needed to modify the plan
tokenizer = AutoTokenizer.from_pretrained('/mnt/liao/planner/models/strans')
model = AutoModel.from_pretrained('/mnt/liao/planner/models/strans')
path = '/mnt/liao/planner/documents/responses_score_reason.txt'
code_rep, code_rep_embd, math_rep, math_rep_embd, search_rep, search_rep_embd, commonsense_rep, commonsense_rep_embd = representations(path)
agent_rep_embd = {'code_agent': code_rep_embd, 'math_agent': math_rep_embd, 'search_agent': search_rep_embd, 'commonsense_agent': commonsense_rep_embd}
agent_rep = {'code_agent': code_rep, 'math_agent': math_rep, 'search_agent': search_rep, 'commonsense_agent': commonsense_rep}

descriptions = {'code_agent': code_agent_descriptions[0],
                'math_agent': math_agent_descriptions[0],
                'search_agent': search_agent_descriptions[0],
                'commonsense_agent': commonsense_agent_descriptions[0]}

MLP = SimilarityMLP()
MLP.load_state_dict(torch.load("reward_model/MLP_high.pt"))

# endregion

# read in the user queries
data_path = '/mnt/liao/planner/datasets/huskyqa'
raw_datasets = datasets.load_dataset(data_path)

planner = planner_gpt()

# region generate new plan with our method
for i in range(237, 246):
    plan = None
    for j in range(5):
        res = planner.plan(raw_datasets['test'][i]['question'])
        print(res)
        if isinstance(res.get('data'), dict):
            if is_valid_json(res['data']['response']['choices'][0]['message']['content']):
                plan = json.loads(res['data']['response']['choices'][0]['message']['content'])
                new_plan = plan_modify(plan, descriptions, MLP, agent_rep, agent_rep_embd)
                with open('documents/new_plan.txt', 'a') as file:
                    file.write('***'+str(i)+': '+raw_datasets['test'][i]['question']+'***\n'+'[*\n'+str(new_plan)+'\n*]\n')
                break
    if plan == None:
        with open('documents/new_plan.txt', 'a') as file:
            file.write('***'+str(i)+': '+raw_datasets['test'][i]['question']+'***\n'+'API error, no plan.')
# endregion


# region collect response


with open('/mnt/liao/planner/documents/new_plan.txt', 'r', encoding='utf-8') as file:
    content = file.read()

pattern_plan = re.compile(r'\[\*\n(.*?)\n\*\]')
matches_plan = pattern_plan.findall(content)
pattern_query = re.compile(r'\*\*\*(.*?)\*\*\*')
matches_query = pattern_query.findall(content)

for i in range(247, len(matches_query)):
    query = matches_query[i]
    plan = json.loads(matches_plan[i])
    query, subtasks_response, final_answer = plan_execution(query, plan)
    with open('/mnt/liao/planner/documents/new_plan_response.txt', 'a', encoding='utf-8') as file:
        file.write('***' + query + '***\n')
        json.dump(subtasks_response, file)
        file.write('\n~~~' + str(i) + ': ' + final_answer + '~~~\n\n')

# endregion

# region compare to get labels
with open('/mnt/liao/planner/documents/new_plan_response.txt', 'r', encoding='utf-8') as file:
    content = file.read()

pattern_query = re.compile(r'\*\*\*(.*?)\*\*\*', re.DOTALL)
matches_query = pattern_query.findall(content)
pattern_answer = re.compile(r'\~\~\~(.*?)\~\~\~', re.DOTALL)
matches_answer = pattern_answer.findall(content)

data_path = '/mnt/liao/planner/datasets/huskyqa'
raw_datasets = datasets.load_dataset(data_path)

from planner import agent
label = []
for num in range(len(matches_query)):
    # print('query: ', matches_query[num])
    # print('ground truth: ', raw_datasets['test']['answer'][num])
    # print('prediction: ', matches_answer[num])
    prompt = evaluate_prompt % (matches_query[num], raw_datasets['test']['answer'][num], matches_answer[num])
    res = agent(prompt)
    print(res['data']['response']['choices'][0]['message']['content'])
    label.append(res['data']['response']['choices'][0]['message']['content'])

# endregion


data_path = '/mnt/liao/planner/datasets/huskyqa'
raw_datasets = datasets.load_dataset(data_path)

for num in range(len(raw_datasets['test']['question'])):
    query = raw_datasets['test']['question'][num]
    for i in range(5):
        res = agent(query, model = 'gpt-4o')
        if isinstance(res['data'], dict):
            gpt4o_answer = res['data']['response']['choices'][0]['message']['content']
            break
    with open('/mnt/liao/planner/documents/gpt4o_response.txt', 'a', encoding='utf-8') as file:
        file.write('***' + str(num) + ': ' + query + '***\n')
        file.write('~~~' + str(num) + ': ' + gpt4o_answer + '~~~\n\n')


data_path = '/mnt/liao/planner/datasets/huskyqa'
raw_datasets = datasets.load_dataset(data_path)

file_path = '/mnt/liao/planner/documents/subtasks.txt'

def is_valid_json(variable):
    try:
        json.loads(variable)
    except ValueError as e:
        return False
    return True

start_marker = '***'
end_marker = '*]'
agent_num_each_subtask = 2

extracted_contents = extract_content_between_markers(file_path, start_marker, end_marker)

queries = []
plans = []
for i in range(len(extracted_contents)):
    split_text = extracted_contents[i].split("***")
    if is_valid_json('['+split_text[1].split('[*')[1]+']'):
        plans.append(split_text[1].split('[*')[1])
        queries.append(split_text[0].strip())

print(len(queries), len(plans))
print(plans[0])


for i in range(len(raw_datasets['test'])):
    query = raw_datasets['test'][i]['question']
    if query in queries:
        idx = queries.index(query)
        with open('all_query_plan.txt', 'a') as file:
            file.write('***'+str(i)+': '+query+'***\n'+'[*'+plans[idx]+'*]\n\n')
    else:
        print('----',i,'----')
        for j in range(5):
            res = planner(query)
            if isinstance(res['data'], dict):
                if is_valid_json(res['data']['response']['choices'][0]['message']['content']):
                    with open('all_query_plan.txt', 'a') as file:
                        file.write('***'+str(i)+': '+query+'***\n'+'[*'+res['data']['response']['choices'][0]['message']['content'][1:-1]+'*]'+'\n')
                        break


# endregion


# region 获得只选择一个agent的结果

with open('/mnt/liao/planner/documents/all_query_plan.txt', 'r', encoding='utf-8') as file:
    content = file.read()

pattern_plan = re.compile(r'\[\*\n(.*?)\n\*\]', re.DOTALL)
matches_plan = pattern_plan.findall(content)
pattern_query = re.compile(r'\*\*\*(.*?)\*\*\*', re.DOTALL)
matches_query = pattern_query.findall(content)

for i in range(len(matches_query)):
    query = matches_query[i]
    plan = json.loads('['+matches_plan[i]+']')
    query, subtasks_response, final_answer = plan_execution(query, plan)
    with open('/mnt/liao/planner/documents/baseline1_choose_one_agent.txt', 'a', encoding='utf-8') as file:
        file.write('***' + query + '***\n')
        json.dump(subtasks_response, file)
        file.write('\n~~~' + str(i) + ': ' + final_answer + '~~~\n\n')



# endregion
with open('/mnt/liao/planner/documents/baseline1_choose_one_agent.txt', 'r', encoding='utf-8') as file:
    content = file.read()
pattern = re.compile(r'\*\*\*(.*?)\*\*\*', re.DOTALL)
matches = pattern.findall(content)
len(matches)





# read in the dataset
with open('/mnt/liao/planner/datasets/huskyqa/final_question.txt', 'r', encoding='utf-8') as file:
    document = file.read()
    
questions = []
for line in document.splitlines():
    line = line.strip()  # 去掉前后的空白符
    questions.append(line)

number_list = []
for i in range(40):
    if i not in [4,10,13,17,21,26,29,35]:
        number_list += range(45*i, 45*(i+1))
random_numbers = sorted(random.sample(number_list, 480))

planner = planner_gpt()
total_time = 0
total_prompt_token = 0
total_completion_token = 0
for i in random_numbers:
    plan = None
    start_time = time.time() 
    for j in range(10):
        res = planner.plan(questions[i])
        print(res)
        if isinstance(res.get('data'), dict):
            if is_valid_json(res['data']['response']['choices'][0]['message']['content']):

                # record plan 
                plan = json.loads(res['data']['response']['choices'][0]['message']['content'])
                with open('/mnt/liao/planner/datasets/huskyqa/final_question_plan.txt', 'a') as file:
                    file.write('***'+str(i)+': '+questions[i]+'***\n'+'[*\n'+str(plan)+'\n*]\n')

                # record consuming token
                # print(res['data']['response']['usage']['prompt_tokens'], res['data']['response']['usage']['completion_tokens'])
                total_prompt_token += res['data']['response']['usage']['prompt_tokens']
                total_completion_token += res['data']['response']['usage']['completion_tokens']

                break

    if plan == None:
        # api error, no plan
        with open('/mnt/liao/planner/datasets/huskyqa/final_question_plan.txt', 'a') as file:
            file.write('***'+str(i)+': '+questions[i]+'***\n'+'API error, no plan.')

    end_time = time.time()
    total_time += end_time - start_time
    print(i,': ', total_time, total_prompt_token, total_completion_token)

with open('/mnt/liao/planner/datasets/huskyqa/final_question_plan.txt', 'r', encoding='utf-8') as file:
    content = file.read()

pattern_plan = re.compile(r'\[\*\n(.*?)\n\*\]')
matches_plan = pattern_plan.findall(content)
pattern_query = re.compile(r'\*\*\*(.*?)\*\*\*')
matches_query = pattern_query.findall(content)

agent_num_each_subtask = 2

for id in range(26, len(matches_query)):
    query = matches_query[id]
    tasks = json.loads(matches_plan[id])
    subtasks_response = []
    for i in range(len(tasks)):
        if i == 0 or tasks[i]['dep']==[]:
            history = None 
        else:
            history = ''
            for j in tasks[i]['dep']:
                scores = [subtasks_response[agent_num_each_subtask*(j-1)+k]['score'] for k in range(agent_num_each_subtask)]
                idx = find_max_position(scores)
                history = history + subtasks_response[agent_num_each_subtask*(j-1)+idx]['response'] + '\n'

        subtask = tasks[i]['task']
        for name in ['name_1', 'name_2']:
            agent_name = tasks[i][name]
            agent_output = get_response(agent_name, subtask, history)
            print(agent_output)
            agent_output['original_query'] = query
        
            if agent_name == 'math_agent':
                response_prompt = '''
                [The Start of the Original Response to Solve the Question]
                %s
                [The End of the Original Response to Solve the Question]
                [The Start of the Rewritten Response]
                %s
                [The End of the Rewritten Response]
                '''
                response = response_prompt % (agent_output['original_answer'], agent_output['response'])
            elif agent_name == 'code_agent':
                response_prompt = '''
                [The Start of the Code to Solve the Question]
                %s
                [The End of the Code to Solve the Question]
                [The Start of the Rewritten Response by Running the Code]
                %s
                [The End of the Rewritten Response by Running the Code]
                '''
                response = response_prompt % (agent_output['code'], agent_output['response'])
            else:
                response = agent_output['response']
            scorer_start_time = time.time()
            for temp in range(10):
                scorer_res = agent(scorer_prompt % (agent_output['agent'], agent_output['task'], response), model = 'gpt-4o')
                print(scorer_res)
                if isinstance(scorer_res.get('data'), dict):
                    agent_output['score_reason'] = scorer_res['data']['response']['choices'][0]['message']['content']
                    print(agent_output['score_reason'])
                    match = re.search(r'Correctness: (\d), Relevance: (\d), Completeness: (\d)', agent_output['score_reason'])
                    if match:
                        correctness = match.group(1)  
                        relevance = match.group(2)     
                        completeness = match.group(3)  
                        if correctness.isdigit() and relevance.isdigit() and completeness.isdigit():
                            correctness = int(correctness)
                            relevance = int(relevance)
                            completeness = int(completeness)
                            scorer_prompt_tokens = scorer_res['data']['response']['usage']['prompt_tokens']
                            scorer_completion_tokens = scorer_res['data']['response']['usage']['completion_tokens']
                            break
                        else:
                            correctness = None
                            relevance = None
                            completeness = None
                    else:
                        correctness = None
                        relevance = None
                        completeness = None
                else:
                    correctness = None
                    relevance = None
                    completeness = None
            scorer_end_time = time.time()
            scorer_time = scorer_end_time - scorer_start_time

            agent_output['correctness'] = correctness
            agent_output['relevance'] = relevance
            agent_output['completeness'] = completeness
            agent_output['scorer_time'] = scorer_time
            agent_output['scorer_prompt_tokens'] = scorer_prompt_tokens
            agent_output['scorer_completion_tokens'] = scorer_completion_tokens
            agent_output['score'] = level_score(correctness, relevance, completeness)

            subtasks_response.append(agent_output)

            agent_output = {key: [value] for key, value in agent_output.items()}

            df = pd.read_excel('/mnt/liao/planner/datasets/huskyqa/plan_execution_score.xlsx')
            agent_output_df = pd.DataFrame(agent_output)
            df = pd.concat([df, agent_output_df], ignore_index=True)
            df.to_excel('/mnt/liao/planner/datasets/huskyqa/plan_execution_score.xlsx', index=False)

    with open('/mnt/liao/planner/datasets/huskyqa/plan_execution_score.txt', 'a') as file:
        file.write('~~~')
        json.dump(subtasks_response, file)
        file.write('~~~'+'\n')

scorer_prompt = """You are an impartial scorer. Given the task and the response from %s, you are responsible for evaluating the quality of the response and providing a score. Please give score to the response given the accuracy (Does the response solve the task? Is the information accurate?), relevance (Is the information in the response directly relevant to the task? Is there any redundant or irrelevant content?) and completeness (Is there any missing necessary information to solve the task?). Begin your evaluation by providing a short explanation. Rate the response from 0 to 10 by strictly following this format: "Score: ***score***", for example: "Score: ***10***". The response like 'I don't know' should be given a low score.
---
[Task]
%s
[The Start of Agent’s Response]
%s
[The End of Agent’s Response]
'''
"""

df = pd.read_excel('/mnt/liao/planner/datasets/huskyqa_train/plan_execution_score.xlsx')

for i in range(len(df)):
    agent_output = {}
    query = df['original_query'][i]
    name = df['agent'][i]
    task = df['task'][i]
    agent_output['original_query'] = query
    agent_output['task'] = task
    agent_output['agent'] = name
    agent_output
    if name == 'math_agent':
        response_prompt = '''
        [The Start of the Original Response to Solve the Question]
        %s
        [The End of the Original Response to Solve the Question]
        [The Start of the Rewritten Response]
        %s
        [The End of the Rewritten Response]
        '''
        response = response_prompt % (df['original_answer'][i], df['response'][i])
        agent_output['original_answer'] = df['original_answer'][i]
        agent_output['response'] = df['response'][i]
    elif name == 'code_agent':
        response_prompt = '''
        [The Start of the Code to Solve the Question]
        %s
        [The End of the Code to Solve the Question]
        [The Start of the Rewritten Response by Running the Code]
        %s
        [The End of the Rewritten Response by Running the Code]
        '''
        response = response_prompt % (df['code'][i], df['response'][i])
        agent_output['code'] = df['code'][i]
        agent_output['response'] = df['response'][i]
    else:
        response = df['response'][i]
        agent_output['response'] = df['response'][i]

    scorer_start_time = time.time()
    for temp in range(10):
        scorer_res = agent(scorer_prompt % (name, task, response), model = 'gpt-4o')
        print(scorer_res)
        if isinstance(scorer_res.get('data'), dict):
            agent_output['score_reason'] = scorer_res['data']['response']['choices'][0]['message']['content']
            print(agent_output['score_reason'])
            match = re.search(r'Score: \*\*\*(\d+)\*\*\*', agent_output['score_reason'])
            if match:
                score_0_10_original = match.group(1)
                if score_0_10_original.isdigit():
                    score_0_10_original = int(score_0_10_original)
                    scorer_prompt_tokens = scorer_res['data']['response']['usage']['prompt_tokens']
                    scorer_completion_tokens = scorer_res['data']['response']['usage']['completion_tokens']
                    break
                else:
                    score_0_10_original = None
            else:
                score_0_10_original = None
        else:
            score_0_10_original = None
    scorer_end_time = time.time()
    scorer_time = scorer_end_time - scorer_start_time
    agent_output['scorer_time'] = scorer_time
    agent_output['scorer_prompt_tokens'] = scorer_prompt_tokens
    agent_output['scorer_completion_tokens'] = scorer_completion_tokens
    agent_output['score'] = score_0_10_original
    agent_output = {key: [value] for key, value in agent_output.items()}

    df_save = pd.read_excel('/mnt/liao/planner/datasets/huskyqa_train/plan_execution_score_0_10_original.xlsx')
    agent_output_df = pd.DataFrame(agent_output)
    df_save = pd.concat([df_save, agent_output_df], ignore_index=True)
    df_save.to_excel('/mnt/liao/planner/datasets/huskyqa_train/plan_execution_score_0_10_original.xlsx', index=False)

# train the reward model with scores in plan_execution_score
df = pd.read_excel('/mnt/liao/planner/datasets/huskyqa_train/plan_execution_score_0_5.xlsx')
subtasks = []
descriptions = []
scores = []
for i in range(len(df)):
    agent_name = df['agent'][i]
    for _ in range(10):
        subtasks.append(df['task'][i])
        scores.append(df['score'][i])
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

x_tensor = torch.cat((subtasks_embd, descriptions_embd), 1)
y_tensor = torch.tensor(scores).reshape(-1, 1)

train_ratio = 0.9
dataset = TensorDataset(x_tensor, y_tensor)

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

torch.save(train_dataset, '/mnt/liao/planner/datasets/huskyqa_train/train_0_5.pt')
torch.save(val_dataset, '/mnt/liao/planner/datasets/huskyqa_train/val_0_5.pt')

train_dataset = torch.load('/mnt/liao/planner/datasets/huskyqa/train_criteria.pt')
val_dataset = torch.load('/mnt/liao/planner/datasets/huskyqa/val_criteria.pt')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# training
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MLP = SimilarityMLP()
MLP.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(MLP.parameters(), lr=0.0001)
num_epochs = 100

num_training_steps = num_epochs * len(train_loader)
progress_bar = tqdm(range(num_training_steps))

loss_train = []
loss_eval = []
for epoch in range(num_epochs):
    MLP.train()
    loss1 = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = MLP(x_batch)
        loss = criterion(outputs, y_batch/10)
        loss.backward()
        optimizer.step()
        progress_bar.update(1)
        loss1 += loss * len(outputs)
    loss_train.append(loss1/len(train_dataset))

    MLP.eval()
    loss2 = 0
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = MLP(x_batch)
        loss = criterion(outputs, y_batch/10)
        loss2 += loss * len(outputs)
        
    loss_eval.append(loss2/len(val_dataset))

# endregion

