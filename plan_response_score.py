import openai
import requests
import json
import datasets
import subprocess
import sys
import os
import re
os.chdir('/mnt/liao/planner')

from utils import simplify_answer, find_max_position, extract_content_between_markers
from search import GoogleSearchAPI, BingSearchAPI
from prompt import search_agent_prompt, code_agent_prompt, math_agent_prompt, commonsense_agent_prompt, rewrite_code_agent_prompt, scorer_prompt, pattern, rewrite_math_agent_prompt
from planner import planner_gpt, agent
from get_response import get_response

data_path = '/mnt/liao/planner/datasets/huskyqa'
raw_datasets = datasets.load_dataset(data_path)

for i in range(291):
    res = planner(raw_datasets['test'][i]['question'])
    with open('subtasks.txt', 'a') as file:
        file.write(str(i)+'\n'+raw_datasets['test'][i]['question']+'\n'+res['data']['response']['choices'][0]['message']['content']+'\n')

# endregion

# region search
input_prompt = search_agent_prompt
subtask = "Determine the population of China and India in 2022."
history = None
prompt = input_prompt % (subtask, history)
print(prompt)
res = agent(prompt)
print(res['data']['response']['choices'][0]['message']['content'])

# start searching
search_query = res['data']['response']['choices'][0]['message']['content']
# browser = GoogleSearchAPI(answer_only=False, top_k=1)
# search_result = browser.search(search_query, use_date=False)
browser = BingSearchAPI()
search_result = browser.search(search_query)
print(search_result)
# endregion

# region code
input_prompt = code_agent_prompt
# subtask = "History: The answer of 'Determine the population of China in 2022' is 1.412B. The answer of 'Determine the population of India in 2022' is 1.417B. Task: Calculate the combined population of China and India in 2022."
# subtask = "Determine population of China in 2021"
# subtask = "Determine the height of the Shanghai World Financial Center in feet."
history = "The answer of 'Determine the population of China' is 1,425,149,003"
subtask = "Calculate 1% of the population of China."
# history = None
prompt = input_prompt % (subtask, history)
print(prompt)
res = agent(prompt, model = 'gpt-4o')
print(res['data']['response']['choices'][0]['message']['content'])

res['data']['response']['choices'][0]['message']['content'].strip()
# endregion

# region run the code
text = res['data']['response']['choices'][0]['message']['content'].strip()
start_index = text.find("```python") + len("```python") 
end_index = text.rfind("```")  

extracted_content = text[start_index:end_index].strip()

print(extracted_content)

result = subprocess.run([sys.executable, "-c", extracted_content], capture_output=True, text=True, timeout=10)
if result.stderr.strip() == "":
    code_exec = simplify_answer(result.stdout, convert_to_str=True).strip()
else:
    code_exec = ""

code_exec
# endregion

# region rewrite the code output
input_prompt = rewrite_code_agent_prompt
prompt = input_prompt % (subtask, extracted_content, code_exec)
print(prompt)
res = agent(prompt, model = 'gpt-3.5-turbo')
print(res['data']['response']['choices'][0]['message']['content'])

# endregion



# region math 

input_prompt = math_agent_prompt
subtask = "Calculate the combined population of China and India in 2022."
# subtask = "Calculate the combined population of China and India in 2022."
# subtask = "Determine the population of China in 2022."
history = None
history = "The answer of 'Determine the population of China in 2022' is 1.412B. The answer of 'Determine the population of India in 2022' is 1.417B."
prompt = input_prompt % (subtask, history)
print(prompt)
res = agent(prompt)
print(res['data']['response']['choices'][0]['message']['content'])

# endregion

def scorer(agent_name, task, answer, model = 'gpt-4o', scorer_prompt = scorer_prompt):
    input_prompt = scorer_prompt
    prompt = input_prompt % (agent_name, task, answer)
    res = agent(prompt, model = model)
    return res['data']['response']['choices'][0]['message']['content']


task = 'Determine the population of China in 2022.'
agent_name = 'search_agent'
answer = 'The estimated population of China in 2022 is 1,425,887,337.'
# answer = 'As a commonsense agent, I do not have access to real-time data or the ability to predict future events. Therefore, I cannot determine the population of China in 2022 without any historical information or additional context.'
response = scorer(task, agent_name, answer)
print(response)

# pattern = r"The score is \*\*\*([\d\.]+)\*\*\*"
pattern = r"\*\*\*([\d\.]+)\*\*\*"

match = re.search(pattern, response)

if match:
    score = match.group(1)
else:
    score = None
print(score)
# endregion


# region get the subtasks and the reponses and scores
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
len(extracted_contents)
print(extracted_contents[0])

for id in range(len(extracted_contents)):
    split_text = extracted_contents[id].split("***")
    question = split_text[0].strip()
    if is_valid_json('['+split_text[1].split('[*')[1]+']'):
        tasks = json.loads('['+split_text[1].split('[*')[1]+']')
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
                agent_output['original_query'] = question
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
                    # response = "original answer: "+agent_output['original_answer']+'\n'+"rewritten answer: "+agent_output['response']
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
                    # response = "code to solve the question: "+agent_output['code']+'\n'+"rewritten answer by running the code: "+agent_output['response']
                else:
                    response = agent_output['response']

                # scorer_response = scorer(agent_output['task'], agent_output['agent'], response)
                scorer_response = scorer(agent_output['agent'], agent_output['task'], response)
                agent_output['score_reason'] = scorer_response
                print(scorer_response)
                match = re.search(pattern, scorer_response)
                if match:
                    score = match.group(1)
                    if score.isdigit():
                        score = float(score)
                    else:
                        score = None
                else:
                    score = None
                agent_output['score'] = score
                subtasks_response.append(agent_output)
    else:
        subtasks_response = [{'original_question': question, 'tasks': None}]
    with open('/mnt/liao/planner/documents/responses.txt', 'a') as file:
        file.write('***')
        json.dump(subtasks_response, file)
        file.write('***'+'\n')
# endregion

# region Test the scorer
def scorer(agent_name, task, answer, model = 'gpt-4o', scorer_prompt = scorer_prompt):
    input_prompt = scorer_prompt
    prompt = input_prompt % (agent_name, task, answer)
    res = agent(prompt, model = model)
    return res['data']['response']['choices'][0]['message']['content']


task = 'Determine how many more people the tour group will interact with in India than in Pakistan.'
agent_name = 'code_agent'
answer = '''
[The Start of the Code to Solve the Question]
%s
[The End of the Code to Solve the Question]
[The Start of the Rewritten Response by Running the Code]
%s
[The End of the Rewritten Response by Running the Code]
'''

code = '# Given data\npopulation_1_percent_india = 14171730  # 1% of India\'s population in 2022\npopulation_1_percent_pakistan = 2358248.62  # 1% of Pakistan\'s population in 2022\n\n# Calculate the difference\ndifference_in_interactions = population_1_percent_india - population_1_percent_pakistan\n\n# Print the result\nprint(f"The tour group will interact with {difference_in_interactions} more people in India than in Pakistan.")'
response = 'The tour group will interact with 11,813,481.38 more people in India than in Pakistan.'
print(answer%(code, response))
print(scorer_prompt%(agent_name, task, answer%(code, response)))
print(scorer(agent_name, task, answer%(code, response)))

# answer = '''
# [The Start of the Original Response to Solve the Question]
# %s
# [The End of the Original Response to Solve the Question]
# [The Start of the Rewritten Response]
# %s
# [The End of the Rewritten Response]
# '''
# agent_name = 'math_agent'
# original_answer = 'To determine how many more people the tour group will interact with in India than in Pakistan, we need to find the difference between the number of people in India and the number of people in Pakistan.\n\nGiven that 1% of the population of India is approximately 14,171,730 people, we can calculate the total population of India by dividing this number by 0.01:\n\nTotal population of India = 14,171,730 / 0.01 = 1,417,173,000\n\nTherefore, the number of people in India is approximately 1,417,173,000.\n\nSimilarly, 1% of the population of Pakistan is approximately 2,358,248.62 people. To find the total population of Pakistan, we divide this number by 0.01:\n\nTotal population of Pakistan = 2,358,248.62 / 0.01 = 235,824,862\n\nTherefore, the number of people in Pakistan is approximately 235,824,862.\n\nTo find the difference between the number of people in India and Pakistan, we subtract the population of Pakistan from the population of India:\n\nDifference = Population of India - Population of Pakistan\nDifference = 1,417,173,000 - 235,824,862\n\nCalculating the difference, we find:\n\nDifference = 1,181,348,138\n\nTherefore, the tour group will interact with approximately 1,181,348,138 more people in India than in Pakistan.\n\nThe answer is \\boxed{1,181,348,138}.'
# response = 'The tour group will interact with approximately 1,181,348,138 more people in India than in Pakistan.'
# print(answer%(original_answer, response))
# print(scorer(agent_name, task, answer%(original_answer, response)))

# input_prompt = scorer_prompt
# prompt = input_prompt % (agent_name, task, answer%(code, response))
# res = agent(prompt)
scorer_response = scorer(agent_name, task, answer%(code, response))
# endregion

# region try to rescore
with open('/mnt/liao/planner/documents/responses.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 使用正则表达式提取 [ {}, {} ] 形式的内容
pattern = re.compile(r'\*\*\*(.*?)\*\*\*')
matches = pattern.findall(content)
# for match in matches:
for i in range(230, len(matches)):
    match = matches[i]
    # match = matches[-1]
    # break
    subtasks_response = []
    match_json = json.loads(match)
    for j in range(len(match_json)):
        # break
        agent_name = match_json[j]['agent']
        task = match_json[j]['task']
        if agent_name == 'math_agent':
            response_prompt = '''
            [The Start of the Original Response to Solve the Question]
            %s
            [The End of the Original Response to Solve the Question]
            [The Start of the Rewritten Response]
            %s
            [The End of the Rewritten Response]
            '''
            response = response_prompt % (match_json[j]['original_answer'], match_json[j]['response'])
        elif agent_name == 'code_agent':
            response_prompt = '''
            [The Start of the Code to Solve the Question]
            %s
            [The End of the Code to Solve the Question]
            [The Start of the Rewritten Response by Running the Code]
            %s
            [The End of the Rewritten Response by Running the Code]
            '''
            response = response_prompt % (match_json[j]['code'], match_json[j]['response'])
            # response = "code to solve the question: "+agent_output['code']+'\n'+"rewritten answer by running the code: "+agent_output['response']
        else:
            response = match_json[j]['response']
        scorer_response = scorer(agent_name, task, response)
        match_json[j]['score_reason'] = scorer_response
        print(scorer_response)
        match = re.search(pattern, scorer_response)
        if match:
            score = match.group(1)
            if score.isdigit():
                score = float(score)
            else:
                score = None
        else:
            score = None
        match_json[j]['score'] = score
        subtasks_response.append(match_json[j])
    with open('/mnt/liao/planner/documents/responses_score_reason.txt', 'a') as file:
        file.write('***')
        json.dump(subtasks_response, file)
        file.write('***'+'\n')
# endregion

# region collect responses from all the agents for each subtask
import copy
names = ['code_agent', 'math_agent', 'search_agent', 'commonsense_agent']
with open('/mnt/liao/planner/documents/responses.txt', 'r', encoding='utf-8') as file:
    content = file.read()

pattern = re.compile(r'\*\*\*(.*?)\*\*\*')
matches = pattern.findall(content)
for i in range(246, len(matches)):
    match = matches[i]
    match_json = json.loads(match)
    all_responses = []
    for i in range(int(len(match_json)/2)):
        temp = copy.deepcopy(match_json[2*i])
        name1 = temp['agent']
        temp.pop('score')
        all_responses.append(temp)
        temp = copy.deepcopy(match_json[2*i+1])
        name2 = temp['agent']
        temp.pop('score')
        all_responses.append(temp)
        history = temp['history']
        subtask = temp['task']
        question = temp['original_query']
        for agent_name in names:
            if agent_name != name1 and agent_name != name2:
                agent_output = get_response(agent_name, subtask, history)
                agent_output['original_query'] = question
                all_responses.append(agent_output)
    with open('/mnt/liao/planner/documents/all_responses_from_each_agent.txt', 'a') as file:
        file.write('***')
        json.dump(all_responses, file)
        file.write('***'+'\n')

# endregion





# region can reward model be used on another dataset?
planner_prompt = """
You are a planning agent. You are responsible for decomposing the given query into subtasks and choose two most suitable agents for each subtask. Your main goal is to efficiently and accurately complete task planning based on the descriptions of agents provided, ensuring the coherence and quality of the subtasks. Please output the subtasks and corresponding agents in the following format: [{"task": task_description, "id": task_id, "name_1": name_of_agent_1, "name_2": name_of_agent_2, "reason": your_detailed_reason_for_the_choice, "dep": dependency_task_ids}]. In this format, "task" is a description of the subtask, which will be used as the input of the chosen agents; "dep" denotes the id of the previous subtask which generates a new resource relied by the current subtask. The available agents and the corresponding descriptions are: [code_agent: Generates code in Python for precise computations to solve the given task. math_agent: Answer math questions by reasoning step-by-step. search_agent: Call Bing Search API for obtaining information regarding the given task. commonsense_agent: Answer the given question with commonsense reasoning.]. Here is an example: \{"query": "If a plane can carry 300 passengers and decides to fly from China to Indonesia, how many full flights are needed to transport 1% of the population of China to Indonesia?", "output":"[\n    {\n        "task": "Determine the population of China.",\n        "id": 1,\n        "name_1": "search_agent",\n        "name_2": "commonsense_agent",\n        "reason": "The search_agent can find the most recent and accurate population data for China, while the commonsense_agent can verify the plausibility of the data.",\n        "dep": []\n    },\n    {\n        "task": "Calculate 1% of the population of China.",\n        "id": 2,\n        "name_1": "math_agent",\n        "name_2": "code_agent",\n        "reason": "The math_agent can reason through the calculation step-by-step, and the code_agent can perform the precise computation.",\n        "dep": [1]\n    },\n    {\n        "task": "Determine the number of full flights needed to transport 1% of the population of China to Indonesia, given that each plane can carry 300 passengers.",\n        "id": 3,\n        "name_1": "math_agent",\n        "name_2": "code_agent",\n        "reason": "The math_agent can reason through the division and rounding process, and the code_agent can perform the precise computation to ensure accuracy.",\n        "dep": [2]\n    }\n]"\}. Given the user's query, output the task plan with the format above directly. 
"""

data_path = '/mnt/liao/planner/datasets/drop'
raw_datasets = datasets.load_dataset(data_path)
planner = planner_gpt()
import random

num = random.randint(0, 199)
print(raw_datasets['test'][num]['question'])
res = planner.plan(raw_datasets['test'][num]['question'])
print(res['data']['response']['choices'][0]['message']['content'])

prompt = commonsense_agent_prompt % (raw_datasets['test'][num]['question'], None)
prompt = search_agent_prompt % (raw_datasets['test'][num]['question'], None)
res = agent(prompt, model = 'gpt-4o')
print(res['data']['response']['choices'][0]['message']['content'])
get_response('search_agent', raw_datasets['test'][num]['question'], None)

# endregion
