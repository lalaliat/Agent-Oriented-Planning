import re
import datasets
from prompt import evaluate_prompt
from planner import agent

path = '/mnt/liao/planner/documents/baseline1_choose_one_agent.txt'

def evaluate(path):
    with open(path, 'r') as file:
        content = file.read()

    pattern_query = re.compile(r'\*\*\*(.*?)\*\*\*', re.DOTALL)
    matches_query = pattern_query.findall(content)
    pattern_answer = re.compile(r'\~\~\~(.*?)\~\~\~', re.DOTALL)
    matches_answer = pattern_answer.findall(content)

    
    data_path = '/mnt/liao/planner/datasets/huskyqa'
    raw_datasets = datasets.load_dataset(data_path)

    label = []
    for num in range(len(matches_query)):
        # print('query: ', matches_query[num])
        # print('ground truth: ', raw_datasets['test']['answer'][num])
        # print('prediction: ', matches_answer[num])
        prompt = evaluate_prompt % (matches_query[num], raw_datasets['test']['answer'][num], matches_answer[num])
        for i in range(10):
            res = agent(prompt)
            if isinstance(res['data'], dict):
                label.append(res['data']['response']['choices'][0]['message']['content'])
                print(res['data']['response']['choices'][0]['message']['content'])
                with open('baseline1_label.txt', 'a') as file:
                    file.write(res['data']['response']['choices'][0]['message']['content']+'\n')
                break
    return label

evaluate(path)

# test

with open(path, 'r') as file:
    content = file.read()

pattern_query = re.compile(r'\*\*\*(.*?)\*\*\*', re.DOTALL)
matches_query = pattern_query.findall(content)
pattern_answer = re.compile(r'\~\~\~(.*?)\~\~\~', re.DOTALL)
matches_answer = pattern_answer.findall(content)


data_path = '/mnt/liao/planner/datasets/huskyqa'
raw_datasets = datasets.load_dataset(data_path)

label = []

evaluate_prompt = '''
You are CompareGPT, a machine to verify the correctness of predictions. Answer with only yes/no.\n\n You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the \"Ground-truth answer\" and the \"Prediction\" to determine whether the prediction correctly answers the question. The prediction may contain extra information, but a correct prediction includes the ground-truth answer. You can answer \"yes\" if the prediction includes the ground-truth answer. You must answer \"no\" if there are any specific details in the ground-truth answer that are not mentioned in the prediction. Note that the error within three decimal places is negligible. By the way, give the reason for the evaluation. Think step by step.
---
Question: %s
Ground-truth answer: %s
Prediction: %s
'''
num = 119
prompt = evaluate_prompt % (matches_query[num], raw_datasets['test']['answer'][num], matches_answer[num])
res = agent(prompt, model = 'gpt-3.5-turbo')
print(res['data']['response']['choices'][0]['message']['content'])
