import torch
import torch.nn as nn

# 定义 MLP 模型
class SimilarityMLP(nn.Module):
    def __init__(self):
        super(SimilarityMLP, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    


class SimilarityMLP_single(nn.Module):
    def __init__(self):
        super(SimilarityMLP_single, self).__init__()
        # self.fc1 = nn.Linear(768, 256)
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
