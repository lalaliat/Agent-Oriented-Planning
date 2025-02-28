from torch.utils.data import DataLoader
from MLP import SimilarityMLP
import os
os.chdir('/mnt/liao/planner')
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm

train_dataset = torch.load('datasets/huskyqa-subtasks/train_high.pt')
val_dataset = torch.load('datasets/huskyqa-subtasks/val_high.pt')


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

# save the model
state_dict = MLP.state_dict()
torch.save(state_dict, '/mnt/liao/planner/reward_model/MLP.pt')
