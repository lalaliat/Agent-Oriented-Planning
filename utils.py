import re
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
import subprocess


def simplify_answer(answer, convert_to_str=False):
    if 'relational' in str(type(answer)):
        return str(answer)
    elif 'numpy' in str(type(answer)):
        if answer.shape == ():
            # scalar value
            answer = round(float(answer), 2)
        else:
            # array value
            answer = round(float(answer[0]), 2)
        return str(answer) if convert_to_str else answer
    elif not answer:
        return "[FAIL]"
    else:
        if type(answer) in [list, tuple]:
            if 'sympy' in str(type(answer[0])):
                try:
                    answer = [round(float(x), 2) for x in answer]
                except Exception:
                    answer = [str(x) for x in answer]
            else:
                answer = [str(x) for x in answer]
            if len(answer) == 1:
                answer = answer[0]
            return answer
        else:
            if 'sympy' in str(type(answer)):
                try:
                    answer = round(float(answer), 2)
                except Exception:
                    answer = str(answer)
                return answer
            elif 'int' in str(type(answer)):
                return str(answer) if convert_to_str else answer
            else:
                try:
                    answer = round(float(answer), 4)
                    return str(answer) if convert_to_str else answer
                except:
                    return str(answer) if convert_to_str else answer
        
def find_max_position(scores):
  
    if all(score is None for score in scores):
        return 0

  
    max_value = float('-inf')
    max_position = 0

  
    for idx, score in enumerate(scores):
        if score is not None and score > max_value:
            max_value = score
            max_position = idx
    
    return max_position

def extract_content_between_markers(file_path, start_marker, end_marker):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to find all content between the specified markers
    pattern = re.compile(re.escape(start_marker) + '(.*?)' + re.escape(end_marker), re.DOTALL)
    
    # Find all matches
    matches = pattern.findall(content)
    
    return matches


# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('/mnt/liao/planner/models/strans')
model = AutoModel.from_pretrained('/mnt/liao/planner/models/strans')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def semb(sentence, model, tokenizer):
    # return the sentences' embeddings
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def query_desc_embd(query, descriptions, model, tokenizer):
    query_embd = semb(query, model, tokenizer)
    descriptions_embd = semb(descriptions, model, tokenizer)
    x_tensor = torch.cat((query_embd, descriptions_embd), 1)
    return x_tensor

def score(query, descriptions, model, tokenizer, MLP):
    x_tensor = query_desc_embd(query, descriptions, model, tokenizer)
    score = MLP(x_tensor)
    return score

def sim(s1, s2, model = model, tokenizer = tokenizer):
  
    embds = semb([s1, s2], model, tokenizer)
    cos_sim = F.cosine_similarity(embds[0].unsqueeze(0), embds[1].unsqueeze(0))
    return cos_sim

def agent_rep(rep, threshold = 0.7, model = model, tokenizer = tokenizer):
  
    rep_sim = [rep[0]]
    for i in range(1, len(rep)):
        flag = 0
        for j in range(len(rep_sim)):
            if sim(rep[i], rep_sim[j], model, tokenizer)>threshold:
                flag = 1
                break
        if flag == 0:
            rep_sim.append(rep[i])
    return rep_sim

def query_subtasks(query, matches):
  
    for match in matches:
        match_json = json.loads(match)
        if match_json[0]['original_query'] in query:
            # print(1)
            return match_json
    else:
        return None
    
def is_valid_json(variable):
    try:
        json.loads(variable)
    except ValueError as e:
        return False
    return True

score_map = {
    (2, 2, 2): 8,
    (2, 1, 2): 7,
    (2, 2, 1): 6,
    (2, 1, 1): 5,
    (1, 2, 2): 4,
    (1, 1, 2): 3,
    (1, 2, 1): 2,
    (1, 1, 1): 1,
}

def level_score(correctness, relevance, completeness):
  
    return score_map.get((correctness, relevance, completeness), 0)


def query_ollama_model(prompt, model_name="qwen2-math"):
  
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            # input=prompt.encode("utf-8"),  
            input=prompt,  
            capture_output=True,
            text=True
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception as e:
        print(f"Error querying model: {e}")
        return None
    
