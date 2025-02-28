from prompt import search_agent_prompt, code_agent_prompt, math_agent_prompt, commonsense_agent_prompt, rewrite_code_agent_prompt, rewrite_math_agent_prompt, summarization_agent_prompt, dep_detect_prompt, majority_vote_prompt
from utils import simplify_answer, query_ollama_model
import subprocess
import importlib
import search
importlib.reload(search)
from search import BingSearchAPI
import planner
importlib.reload(planner)
from planner import agent
import sys
import time
import re
from utils import semb
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('/mnt/liao/planner/models/strans')
model = AutoModel.from_pretrained('/mnt/liao/planner/models/strans')
import torch.nn.functional as F
import torch


def get_response(agent_name, subtask, query, ollama = False, history = None, model_name = None):
    if agent_name == 'math_agent':
        if ollama == False:
            start_time = time.time() 
            input_prompt = math_agent_prompt
            prompt = input_prompt % (query, subtask, history)
            for i in range(10):
                res = agent(prompt)
                if isinstance(res.get('data'), dict):
                    original_answer = res['data']['response']['choices'][0]['message']['content']
                    # record comsuming tokens of original answer
                    original_answer_prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                    original_answer_completion_tokens = res['data']['response']['usage']['completion_tokens']
                    break
            print("math agent's original_answer:", original_answer)
        else:
            start_time = time.time()
            input_prompt = math_agent_prompt
            prompt = input_prompt % (query, subtask, history)
            # original_answer = query_ollama_model(prompt, model_name=ollama_name)
            original_answer = query_ollama_model(prompt, 'qwen2-math')
            original_answer_prompt_tokens = 0
            original_answer_completion_tokens = 0
        # rewrite the answer
        input_prompt = rewrite_math_agent_prompt
        prompt = input_prompt % (subtask, original_answer.strip())
        for i in range(10):
            res = agent(prompt)
            if isinstance(res.get('data'), dict):
                rewrite_answer = res['data']['response']['choices'][0]['message']['content']
                # record comsuming tokens of rewritten answer
                rewrite_answer_prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                rewrite_answer_completion_tokens = res['data']['response']['usage']['completion_tokens']
                break
        print("math agent's rewrite_answer:", rewrite_answer)
        # collect the original answer as well for scoring
        end_time = time.time()
        ans = {"task":subtask, "agent": agent_name, "original_answer":original_answer, "response":rewrite_answer, 
               "prompt_tokens": original_answer_prompt_tokens + rewrite_answer_prompt_tokens,
               "completion_tokens": original_answer_completion_tokens + rewrite_answer_completion_tokens,
               "time": end_time - start_time}
    elif agent_name == 'commonsense_agent':
        start_time = time.time()
        input_prompt = commonsense_agent_prompt
        prompt = input_prompt % (query, subtask, history)
        for i in range(10):
            res = agent(prompt)
            if isinstance(res.get('data'), dict):
                prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                completion_tokens = res['data']['response']['usage']['completion_tokens']
                break
        end_time = time.time()
        ans = {"task":subtask, "agent": agent_name, "response":res['data']['response']['choices'][0]['message']['content'],
                       "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "time": end_time - start_time} 
    elif agent_name == 'code_agent':
        if ollama == False:
            # get code
            start_time = time.time()
            input_prompt = code_agent_prompt
            prompt = input_prompt % (query, subtask, history)
            for i in range(10):
                res = agent(prompt)
                if isinstance(res.get('data'), dict):
                    original_answer = res['data']['response']['choices'][0]['message']['content'].strip()
                    original_answer_prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                    original_answer_completion_tokens = res['data']['response']['usage']['completion_tokens']
                    break
        else:
            start_time = time.time()
            # if history is None:
            #     prompt = "Write Python code to solve the problem: " + subtask + "\nThe code should be in the following format: \n```python\n\n```. \nMake sure that the code could be executed directly without needing extra input."
            # else:
            #     prompt = "Write Python code to solve the problem: " + subtask + "\nHistory: " + history + "\nThe code should be in the following format: \n```python\n\n```.\nMake sure that the code could be executed directly without needing extra input."
            input_prompt = code_agent_prompt
            prompt = input_prompt % (query, subtask, history)
            # original_answer = query_ollama_model(prompt, ollama_name)
            original_answer = query_ollama_model(prompt, "deepseek-coder-v2")
            original_answer_prompt_tokens = 0
            original_answer_completion_tokens = 0
            print(original_answer)
        start_index = original_answer.find("```python") + len("```python")  # 找到第一个三引号，并且加3以排除三引号本身
        end_index = original_answer.rfind("```")
        code = original_answer[start_index:end_index].strip()
        print("code agent's code:", code)
        # run the code
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        if result.stderr.strip() == "":
            code_exec = simplify_answer(result.stdout, convert_to_str=True).strip()
        else:
            code_exec = ""
        print(code_exec)
        # rewrite the answer
        input_prompt = rewrite_code_agent_prompt
        prompt = input_prompt % (subtask, code, code_exec)
        for i in range(10):
            res = agent(prompt)
            if isinstance(res.get('data'), dict):
                rewrite_answer = res['data']['response']['choices'][0]['message']['content']
                rewrite_answer_prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                rewrite_answer_completion_tokens = res['data']['response']['usage']['completion_tokens']
                break
        print("code agent's rewrite_answer:", rewrite_answer)
        end_time = time.time()
        ans = {"task":subtask, "agent": agent_name, "code":code, "response":rewrite_answer,
               "prompt_tokens": original_answer_prompt_tokens + rewrite_answer_prompt_tokens,
               "completion_tokens": original_answer_completion_tokens + rewrite_answer_completion_tokens,
               "time": end_time - start_time}
    elif agent_name == 'search_agent':
        start_time = time.time()
        input_prompt = search_agent_prompt
        prompt = input_prompt % (subtask, history)
        for i in range(10):
            res = agent(prompt)
            if isinstance(res.get('data'), dict):
                search_query = res['data']['response']['choices'][0]['message']['content']
                prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                completion_tokens = res['data']['response']['usage']['completion_tokens']
                break
        print(search_query)
        # Call the API
        browser = BingSearchAPI()
        search_result, rewrite_answer_prompt_tokens, rewrite_answer_completion_tokens = browser.search(search_query)
        # print(search_result)
        end_time = time.time()
        ans = {"task":subtask, "agent": agent_name, "response":search_result, 
               "prompt_tokens": prompt_tokens + rewrite_answer_prompt_tokens, "completion_tokens": completion_tokens + rewrite_answer_completion_tokens,
               "time": end_time - start_time}
    return ans

# history = "The population of China in 2022 is 1,411.75 million. India's population in 2022 is estimated to be 1,417,173,000 people."

def plan_execution(query, plan, ollama = False, dep = 'all'):
    history = [] # 只收集短回复
    subtasks_response = [] # 收集完整的plan
    for i in range(len(plan)):

        if type(plan[i]) == list:
            # history_sub = []
            history.append([])
            subtasks_response_sub = []
            for j in range(len(plan[i])):
                agent_name = plan[i][j]['agent'] if 'agent' in plan[i][j] else plan[i][j]['name_1']
                task = plan[i][j]['task']
                # TODO: this part need to be further modified
                if dep == 'all':
                    history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in history], []))
                elif dep == 'dep':
                    history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in history], []))
                elif dep == 'detect':
                    history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in history], []))
                agent_output = get_response(agent_name, task, query, ollama, history_provide)
                print(agent_name, ': ', agent_output)
                history[-1].append(agent_output['response'])
                subtasks_response_sub.append(agent_output)
            # history.append(history_sub)
            subtasks_response.append(subtasks_response_sub)
        else:
            agent_name = plan[i]['agent'] if 'agent' in plan[i] else plan[i]['name_1']
            task = plan[i]['task']
            print(task)
            if dep == 'all':
                history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in history], []))
            elif dep == 'dep':
                history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in [history[temp-1] for temp in plan[i]['dep']]], []))
            elif dep == 'detect':
                if agent_name == 'search_agent' or agent_name == 'commonsense_agent':
                    history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in [history[temp-1] for temp in plan[i]['dep']]], []))
                else:
                    history_ask = '\n'.join(sum([x if isinstance(x, list) else [x] for x in [history[temp-1] for temp in plan[i]['dep']]], []))
                    # history_total = '\n'.join(sum([x if isinstance(x, list) else [x] for x in history], []))
                    history_total = '\n'.join(f"{i+1}. {x}" for i, x in enumerate(history))
                    history_provide = history_ask
                    prompt = dep_detect_prompt % (task, history_ask, history_total, query)
                    for temp in range(10):
                        res = agent(prompt)
                        if isinstance(res.get('data'), dict):
                            answer1 = re.compile(r'\*\*\*(.*?)\*\*\*').findall(res['data']['response']['choices'][0]['message']['content'])
                            # 首先第一个问题必须有回复
                            if answer1 != []:
                                if answer1[0] == 'Yes':
                                    history_provide = history_ask
                                    break
                                elif answer1[0] == 'No':
                                    # 如果第一个问题回答No，那么第二个问题必须有回复
                                    answer2 = re.compile(r'\~\~\~(.*?)\~\~\~').findall(res['data']['response']['choices'][0]['message']['content'])
                                    answer3 = re.compile(r'\$\$\$(.*?)\$\$\$').findall(res['data']['response']['choices'][0]['message']['content'])
                                    if answer3 !=[] and answer2 != []:
                                        # task_add 要么是None，要么是需要补充的子任务
                                        # dep_add 要么是None，要么是需要补充的依赖，为list
                                        dep_add = answer2[0]
                                        task_add = answer3[0]
                                        if task_add == 'None':
                                            task_add = None
                                        if dep_add == 'None':
                                            dep_add = None
                                        else:
                                            dep_add = list(map(int, dep_add.split(',')))
                                        if dep_add != None:
                                            history_provide += '\n'.join([history[i-1] for i in dep_add])
                                        if task_add != None:
                                            # 其实应该replan重新调用agent，但是实在是绷不住了
                                            for _ in range(10):
                                                res = agent(task_add)
                                                if isinstance(res.get('data'), dict):
                                                    history_provide += res['data']['response']['choices'][0]['message']['content']
                                                    break
                                        break
            print('history: ', history_provide)
            agent_output = get_response(agent_name, task, query, ollama, history_provide)
            print(agent_name, ': ', agent_output)
            history.append(agent_output['response'])
            subtasks_response.append(agent_output)

    # extract the plan and the corresponding responses
    extract_plan = []
    extract_responses = []
    for i in range(len(subtasks_response)):
        extract_plan_sub = []
        extract_responses_sub = []
        if type(subtasks_response[i]) == list:
            for j in range(len(subtasks_response[i])):
                extract_plan_sub.append(subtasks_response[i][j]['task'])
                extract_responses_sub.append(subtasks_response[i][j]['response'])
            extract_plan.append(extract_plan_sub)
            extract_responses.append(extract_responses_sub)
        else:
            extract_plan.append(subtasks_response[i]['task'])
            extract_responses.append(subtasks_response[i]['response'])

    # provide the information above to the summarization agent.
    prompt = summarization_agent_prompt % (query, extract_plan, extract_responses)
    for i in range(10):
        res = agent(prompt)
        if isinstance(res, dict) and isinstance(res.get('data'), dict):
            final_answer = res['data']['response']['choices'][0]['message']['content']
            break
    return query, subtasks_response, final_answer



def plan_execution_multi_math(query, plan, gpt_3_5_rep, gpt_4o_rep, qwen_rep, llama_rep, ollama = False, dep = 'all', math_threshold = 0.5):
    history = [] # 只收集短回复
    subtasks_response = [] # 收集完整的plan
    for i in range(len(plan)):
        if type(plan[i]) == list:
            # history_sub = []
            history.append([])
            subtasks_response_sub = []
            for j in range(len(plan[i])):
                agent_name = plan[i][j]['agent'] if 'agent' in plan[i][j] else plan[i][j]['name_1']
                task = plan[i][j]['task']
                history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in history], []))

                if agent_name == 'math_agent':
                    task_embd = semb(task, model, tokenizer)
                    gpt_3_5_rep_sim = torch.max(F.cosine_similarity(task_embd, semb(gpt_3_5_rep, model, tokenizer)))
                    gpt_4o_rep_sim = torch.max(F.cosine_similarity(task_embd, semb(gpt_4o_rep, model, tokenizer)))
                    qwen_rep_sim = torch.max(F.cosine_similarity(task_embd, semb(qwen_rep, model, tokenizer)))
                    llama_rep_sim = torch.max(F.cosine_similarity(task_embd, semb(llama_rep, model, tokenizer)))
                    sim = [gpt_3_5_rep_sim, gpt_4o_rep_sim, qwen_rep_sim, llama_rep_sim]
                    model_names = ['gpt_3_5', 'gpt_4o', 'qwen', 'llama']
                    if max(sim) >= math_threshold:
                        model_name = model_names[sim.index(max(sim))]
                        agent_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, model_name)
                    else:
                        # 通过majority vote来决定使用哪个结果
                        gpt_3_5_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, 'gpt_3_5')
                        gpt_4o_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, 'gpt_4o')
                        qwen_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, 'qwen')
                        llama_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, 'llama')
                        prompt_output = majority_vote_prompt % (task, gpt_3_5_output['response'], gpt_4o_output['response'], qwen_output['response'], llama_output['response'])
                        for _ in range(10):
                            res = agent(prompt_output)
                            if isinstance(res.get('data'), dict):
                                temp_output = res['data']['response']['choices'][0]['message']['content']
                                if re.findall(r'<SUPPORTING_AGENTS>(.*?)</SUPPORTING_AGENTS>', temp_output, re.DOTALL) == []:
                                    best_num = [2]
                                else:
                                    temp_num = re.findall(r'<SUPPORTING_AGENTS>(.*?)</SUPPORTING_AGENTS>', temp_output, re.DOTALL)[0].strip()
                                    best_num = list(map(int, temp_num.split(',')))
                                agent_output = [gpt_3_5_output, gpt_4o_output, qwen_output, llama_output][best_num[0]-1]
                                break
                        # 为被选中的agent更新代表作集合
                        if 1 in best_num:
                            gpt_3_5_rep.append(task)
                        if 2 in best_num:
                            gpt_4o_rep.append(task)
                        if 3 in best_num:
                            qwen_rep.append(task)
                        if 4 in best_num:
                            llama_rep.append(task)
                else:
                    agent_output = get_response_multi_math(agent_name, task, query, ollama, history_provide)
                print(agent_name, ': ', agent_output)
                history[-1].append(agent_output['response'])
                subtasks_response_sub.append(agent_output)
            # history.append(history_sub)
            subtasks_response.append(subtasks_response_sub)
        else:
            agent_name = plan[i]['agent'] if 'agent' in plan[i] else plan[i]['name_1']
            task = plan[i]['task']
            if dep == 'all':
                history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in history], []))
            elif dep == 'dep':
                history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in [history[temp-1] for temp in plan[i]['dep']]], []))
            elif dep == 'detect':
                if agent_name == 'search_agent':
                    history_provide = '\n'.join(sum([x if isinstance(x, list) else [x] for x in [history[temp-1] for temp in plan[i]['dep']]], []))
                else:
                    history_ask = '\n'.join(sum([x if isinstance(x, list) else [x] for x in [history[temp-1] for temp in plan[i]['dep']]], []))
                    history_total = '\n'.join(sum([x if isinstance(x, list) else [x] for x in history], []))
                    prompt = dep_detect_prompt % (task, history_ask, history_total)
                    for temp in range(10):
                        res = agent(prompt)
                        if isinstance(res.get('data'), dict):
                            if re.compile(r'\*\*\*(.*?)\*\*\*').findall(res['data']['response']['choices'][0]['message']['content']) == []:
                                label = 'Yes'
                            else: 
                                label = re.compile(r'\*\*\*(.*?)\*\*\*').findall(res['data']['response']['choices'][0]['message']['content'])[0]
                            break
                    if label == 'Yes':
                        history_provide = history_ask
                    elif label == 'Have':
                        history_provide = history_total
                    elif label == 'No':
                        pass
            print('history: ', history_provide)
            if agent_name == 'math_agent':
                task_embd = semb(task, model, tokenizer)
                gpt_3_5_rep_sim = torch.max(F.cosine_similarity(task_embd, semb(gpt_3_5_rep, model, tokenizer)))
                gpt_4o_rep_sim = torch.max(F.cosine_similarity(task_embd, semb(gpt_4o_rep, model, tokenizer)))
                qwen_rep_sim = torch.max(F.cosine_similarity(task_embd, semb(qwen_rep, model, tokenizer)))
                llama_rep_sim = torch.max(F.cosine_similarity(task_embd, semb(llama_rep, model, tokenizer)))
                sim = [gpt_3_5_rep_sim, gpt_4o_rep_sim, qwen_rep_sim, llama_rep_sim]
                model_names = ['gpt_3_5', 'gpt_4o', 'qwen', 'llama']
                if max(sim) >= math_threshold:
                    model_name = model_names[sim.index(max(sim))]
                    agent_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, model_name)
                else:
                    # 通过majority vote来决定使用哪个结果
                    gpt_3_5_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, 'gpt_3_5')
                    gpt_4o_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, 'gpt_4o')
                    qwen_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, 'qwen')
                    llama_output = get_response_multi_math(agent_name, task, query, ollama, history_provide, 'llama')
                    prompt_output = majority_vote_prompt % (task, gpt_3_5_output['response'], gpt_4o_output['response'], qwen_output['response'], llama_output['response'])
                    for _ in range(10):
                        res = agent(prompt_output)
                        if isinstance(res.get('data'), dict):
                            temp_output = res['data']['response']['choices'][0]['message']['content']
                            if re.findall(r'<SUPPORTING_AGENTS>(.*?)</SUPPORTING_AGENTS>', temp_output, re.DOTALL) == []:
                                best_num = [2]
                            else:
                                temp_num = re.findall(r'<SUPPORTING_AGENTS>(.*?)</SUPPORTING_AGENTS>', temp_output, re.DOTALL)[0].strip()
                                best_num = list(map(int, temp_num.split(',')))
                            agent_output = [gpt_3_5_output, gpt_4o_output, qwen_output, llama_output][best_num[0]-1]
                            break
                    # 为被选中的agent更新代表作集合
                    if 1 in best_num:
                        gpt_3_5_rep.append(task)
                    if 2 in best_num:
                        gpt_4o_rep.append(task)
                    if 3 in best_num:
                        qwen_rep.append(task)
                    if 4 in best_num:
                        llama_rep.append(task)
            else:
                agent_output = get_response_multi_math(agent_name, task, query, ollama, history_provide)
            print(agent_name, ': ', agent_output)
            history.append(agent_output['response'])
            subtasks_response.append(agent_output)

    # extract the plan and the corresponding responses
    extract_plan = []
    extract_responses = []
    for i in range(len(subtasks_response)):
        extract_plan_sub = []
        extract_responses_sub = []
        if type(subtasks_response[i]) == list:
            for j in range(len(subtasks_response[i])):
                extract_plan_sub.append(subtasks_response[i][j]['task'])
                extract_responses_sub.append(subtasks_response[i][j]['response'])
            extract_plan.append(extract_plan_sub)
            extract_responses.append(extract_responses_sub)
        else:
            extract_plan.append(subtasks_response[i]['task'])
            extract_responses.append(subtasks_response[i]['response'])

    # provide the information above to the summarization agent.
    prompt = summarization_agent_prompt % (query, extract_plan, extract_responses)
    for i in range(10):
        res = agent(prompt)
        if isinstance(res, dict) and isinstance(res.get('data'), dict):
            final_answer = res['data']['response']['choices'][0]['message']['content']
            break
    return query, subtasks_response, final_answer, gpt_3_5_rep, gpt_4o_rep, qwen_rep, llama_rep


def get_response_multi_math(agent_name, subtask, query, ollama = False, history = None, model_name = None):
    if agent_name == 'math_agent':

        if model_name == 'gpt_3_5' or model_name == 'gpt_4o':
            start_time = time.time() 
            input_prompt = math_agent_prompt
            prompt = input_prompt % (query, subtask, history)
            for i in range(10):
                if model_name =='gpt_3_5':
                    res = agent(prompt, model = 'gpt-3.5-turbo')
                elif model_name == 'gpt_4o':
                    res = agent(prompt, model = 'gpt-4o')
                if isinstance(res.get('data'), dict):
                    original_answer = res['data']['response']['choices'][0]['message']['content']
                    # record comsuming tokens of original answer
                    original_answer_prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                    original_answer_completion_tokens = res['data']['response']['usage']['completion_tokens']
                    break
            print("math agent's original_answer:", original_answer)
        else:
            start_time = time.time()
            input_prompt = math_agent_prompt
            prompt = input_prompt % (query, subtask, history)
            # original_answer = query_ollama_model(prompt, model_name=ollama_name)
            if model_name == 'qwen':
                original_answer = query_ollama_model(prompt, 'qwen2-math')
            elif model_name == 'llama':
                original_answer = query_ollama_model(prompt, 'llama3.2')
            original_answer_prompt_tokens = 0
            original_answer_completion_tokens = 0
        # rewrite the answer
        input_prompt = rewrite_math_agent_prompt
        prompt = input_prompt % (subtask, original_answer.strip())
        for i in range(10):
            res = agent(prompt)
            if isinstance(res.get('data'), dict):
                rewrite_answer = res['data']['response']['choices'][0]['message']['content']
                # record comsuming tokens of rewritten answer
                rewrite_answer_prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                rewrite_answer_completion_tokens = res['data']['response']['usage']['completion_tokens']
                break
        print("math agent's rewrite_answer:", rewrite_answer)
        # collect the original answer as well for scoring
        end_time = time.time()
        ans = {"task":subtask, "agent": agent_name, "original_answer":original_answer, "response":rewrite_answer, 
               "prompt_tokens": original_answer_prompt_tokens + rewrite_answer_prompt_tokens,
               "completion_tokens": original_answer_completion_tokens + rewrite_answer_completion_tokens,
               "time": end_time - start_time}
    elif agent_name == 'commonsense_agent':
        start_time = time.time()
        input_prompt = commonsense_agent_prompt
        prompt = input_prompt % (query, subtask, history)
        for i in range(10):
            res = agent(prompt)
            if isinstance(res.get('data'), dict):
                prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                completion_tokens = res['data']['response']['usage']['completion_tokens']
                break
        end_time = time.time()
        ans = {"task":subtask, "agent": agent_name, "response":res['data']['response']['choices'][0]['message']['content'],
                       "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "time": end_time - start_time} 
    elif agent_name == 'code_agent':
        if ollama == False:
            # get code
            start_time = time.time()
            input_prompt = code_agent_prompt
            prompt = input_prompt % (query, subtask, history)
            for i in range(10):
                res = agent(prompt)
                if isinstance(res.get('data'), dict):
                    original_answer = res['data']['response']['choices'][0]['message']['content'].strip()
                    original_answer_prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                    original_answer_completion_tokens = res['data']['response']['usage']['completion_tokens']
                    break
        else:
            start_time = time.time()
            # if history is None:
            #     prompt = "Write Python code to solve the problem: " + subtask + "\nThe code should be in the following format: \n```python\n\n```. \nMake sure that the code could be executed directly without needing extra input."
            # else:
            #     prompt = "Write Python code to solve the problem: " + subtask + "\nHistory: " + history + "\nThe code should be in the following format: \n```python\n\n```.\nMake sure that the code could be executed directly without needing extra input."
            input_prompt = code_agent_prompt
            prompt = input_prompt % (query, subtask, history)
            # original_answer = query_ollama_model(prompt, ollama_name)
            original_answer = query_ollama_model(prompt, "deepseek-coder-v2")
            original_answer_prompt_tokens = 0
            original_answer_completion_tokens = 0
            print(original_answer)
        start_index = original_answer.find("```python") + len("```python")  # 找到第一个三引号，并且加3以排除三引号本身
        end_index = original_answer.rfind("```")
        code = original_answer[start_index:end_index].strip()
        print("code agent's code:", code)
        # run the code
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        if result.stderr.strip() == "":
            code_exec = simplify_answer(result.stdout, convert_to_str=True).strip()
        else:
            code_exec = ""
        print(code_exec)
        # rewrite the answer
        input_prompt = rewrite_code_agent_prompt
        prompt = input_prompt % (subtask, code, code_exec)
        for i in range(10):
            res = agent(prompt)
            if isinstance(res.get('data'), dict):
                rewrite_answer = res['data']['response']['choices'][0]['message']['content']
                rewrite_answer_prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                rewrite_answer_completion_tokens = res['data']['response']['usage']['completion_tokens']
                break
        print("code agent's rewrite_answer:", rewrite_answer)
        end_time = time.time()
        ans = {"task":subtask, "agent": agent_name, "code":code, "response":rewrite_answer,
               "prompt_tokens": original_answer_prompt_tokens + rewrite_answer_prompt_tokens,
               "completion_tokens": original_answer_completion_tokens + rewrite_answer_completion_tokens,
               "time": end_time - start_time}
    elif agent_name == 'search_agent':
        start_time = time.time()
        input_prompt = search_agent_prompt
        prompt = input_prompt % (subtask, history)
        for i in range(10):
            res = agent(prompt)
            if isinstance(res.get('data'), dict):
                search_query = res['data']['response']['choices'][0]['message']['content']
                prompt_tokens = res['data']['response']['usage']['prompt_tokens']
                completion_tokens = res['data']['response']['usage']['completion_tokens']
                break
        print(search_query)
        # Call the API
        browser = BingSearchAPI()
        search_result, rewrite_answer_prompt_tokens, rewrite_answer_completion_tokens = browser.search(search_query)
        # print(search_result)
        end_time = time.time()
        ans = {"task":subtask, "agent": agent_name, "response":search_result, 
               "prompt_tokens": prompt_tokens + rewrite_answer_prompt_tokens, "completion_tokens": completion_tokens + rewrite_answer_completion_tokens,
               "time": end_time - start_time}
    return ans
