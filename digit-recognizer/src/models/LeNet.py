import sys
sys.path.append(r"/workspace/kaggle/digit-recognizer/src/data")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_process import train_dataloader, valid_dataloader, test_dataloader

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),nn.Sigmoid(),
            nn.Linear(120,84),nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        return self.net(x)
    
model = LeNet().to("cuda")
epochs = 10
lr = 0.9
loss_fn = nn.CrossEntropyLoss()
losses = []
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
    