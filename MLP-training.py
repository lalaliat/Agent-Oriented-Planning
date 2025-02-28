import re
import json
import matplotlib.pyplot as plt
import os
os.chdir('/mnt/liao/planner')
from embedding_model import semb
from MLP import SimilarityMLP
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim


with open('/mnt/liao/planner/documents/responses.txt', 'r', encoding='utf-8') as file:
    content = file.read()

pattern = re.compile(r'\*\*\*(.*?)\*\*\*')
matches = pattern.findall(content)

for match in matches:
    print(match)
    break

match_json = json.loads(match)


scores = []
flag = 0
for match in matches:
    match_json = json.loads(match)
    score = []
    for j in range(len(match_json)):
        if 'score' not in match_json[j]:
            score = []
        else:
            score.append(match_json[j]['score'])
    scores.append(score)
    flag+=1

plt.hist(scores)
plt.show()

MLP = SimilarityMLP()

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('/mnt/liao/planner/models/strans')
model = AutoModel.from_pretrained('/mnt/liao/planner/models/strans')

agents_descriptions = {'code_agent': 'Generates code in Python for precise computations to solve the given task.',
          'math_agent': 'Answer math questions by reasoning step-by-step.',
          'search_agent': 'Write a concise, informative Bing Search query for obtaining information regarding the given task.',
          'commonsense_agent': 'Performs commonsense reasoning.'}

with open('/mnt/liao/planner/responses.txt', 'r', encoding='utf-8') as file:
    content = file.read()

pattern = re.compile(r'\*\*\*(.*?)\*\*\*')
matches = pattern.findall(content)

# extract subtasks, descriptions, and scores
subtasks = []
descriptions = []
scores = []
for match in matches:
    # break
    match_json = json.loads(match)
    for j in range(len(match_json)):
        scores.append(match_json[j]['score'])
        subtasks.append(match_json[j]['task'])
        agent_name = match_json[j]['agent']
        descriptions.append(agents_descriptions[agent_name])

subtasks_embd = semb(subtasks, model, tokenizer) 
descriptions_embd = semb(descriptions, model, tokenizer)

x_tensor = torch.cat((subtasks_embd, descriptions_embd), 1)
y_tensor = torch.tensor(scores).reshape(-1, 1)

dataset = TensorDataset(x_tensor, y_tensor)

train_ratio = 0.9

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset = torch.load('/mnt/liao/planner/datasets/huskyqa-subtasks/train_original.pt')
val_dataset = torch.load('/mnt/liao/planner/datasets/huskyqa-subtasks/val_original.pt')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

MLP = SimilarityMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(MLP.parameters(), lr=0.0001)

num_epochs = 10
losses = []
for epoch in range(num_epochs):
    MLP.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = MLP(x_batch)
        loss = criterion(outputs, y_batch/10)
        losses.append(loss)
        loss.backward()
        optimizer.step()

MLP.eval()
val_losses = []
for x_batch, y_batch in val_loader:
    outputs = MLP(x_batch)
    loss = criterion(outputs, y_batch/10)
    val_losses.append(loss)

val_losses
# b = copy.deepcopy(MLP.state_dict())

# for name, param in a.items():
#     print(torch.sum(abs(a[name]-b[name])))
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
writer.flush()



# change the descriotions to generate more training datasets.
code_agent_descriptions = [
    "Generates code in Python for precise computations to solve the given task.",
    "Generates Python code for precise computations to resolve the task at hand.",
    "Produces Python code for exact computations to complete the given task.",
    "Develops Python code to perform precise computations for the given task.",
    "Writes Python code to carry out exact calculations to solve the provided task.",
    "Crafts Python code to achieve accurate computations for the assigned task.",
    "Constructs Python programs for precise calculations to accomplish the given task.",
    "Develops scripts in Python to execute precise computations to tackle the given task.",
    "Produces Python code for accurate computations to address the task at hand.",
    "Crafts Python code for exact computations to resolve the designated task."
]

math_agent_descriptions = [
    "Answer math questions by reasoning step-by-step.",
    "Solve math problems through step-by-step reasoning.",
    "Approach math questions using a step-by-step reasoning method.",
    "Tackle math problems by reasoning out each step.",
    "Work through math questions with a step-by-step reasoning process.",
    "Answer math problems by reasoning each step one at a time.",
    "Resolve math questions using a methodical, step-by-step approach.",
    "Solve math questions via a step-by-step reasoning approach.",
    "Handle math problems by following a step-by-step reasoning strategy.",
    "Work on math questions by applying reasoning step-by-step."
]

search_agent_descriptions = [
    "Call Bing Search API for obtaining information regarding the given task.",
    "Use the Bing Search API to get information about the specified task.",
    "Call Bing Search API to retrieve information related to the designated task.",
    "Harness Bing Search API to find information regarding the specified task.",
    "Use Bing Search API to obtain necessary information about the given task.",
    "Engage with Bing Search API to retrieve details regarding the specified task.",
    "Apply Bing Search API for obtaining data about the described task.",
    "Call upon Bing Search API to access information about the given task.",
    "Utilize the Bing Search API to source information on the specified task.",
    "Employ the Bing Search API to acquire the necessary information on the given task."
]

commonsense_agent_descriptions =  [
    "Answer the given question with commonsense reasoning.",
    "Respond to the provided question using commonsense reasoning.",
    "Use commonsense reasoning to answer the question given.",
    "Apply commonsense reasoning to the stated question.",
    "Answer the question presented with commonsense reasoning.",
    "Answer the specified question with commonsense reasoning.",
    "Utilize commonsense reasoning to answer the provided question.",
    "Use commonsense reasoning to respond to the supplied question.",
    "Address the question with commonsense reasoning.",
    "Employ commonsense reasoning to answer the question given.",
]
with open('/mnt/liao/planner/documents/responses.txt', 'r', encoding='utf-8') as file:
    content = file.read()

pattern = re.compile(r'\*\*\*(.*?)\*\*\*')
matches = pattern.findall(content)

# extract subtasks, descriptions, and scores
subtasks = []
descriptions = []
scores = []
for match in matches:
    # break
    match_json = json.loads(match)
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

subtasks_embd = semb(subtasks, model, tokenizer) 
descriptions_embd = semb(descriptions, model, tokenizer)

x_tensor = torch.cat((subtasks_embd, descriptions_embd), 1)
y_tensor = torch.tensor(scores).reshape(-1, 1)

dataset = TensorDataset(x_tensor, y_tensor)

train_ratio = 0.9

train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# endregion



# region training
from tqdm.autonotebook import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MLP = SimilarityMLP()
MLP.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(MLP.parameters(), lr=0.0001)
num_epochs = 100

num_training_steps = num_epochs * len(train_loader)
progress_bar = tqdm(range(num_training_steps))
# a = copy.deepcopy(MLP.state_dict())


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
