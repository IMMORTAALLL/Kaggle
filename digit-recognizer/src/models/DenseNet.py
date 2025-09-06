import sys
sys.path.append(r"kaggle/digit-recognizer/src/data")

import os
import torch
import pandas
import torch.nn as nn
import matplotlib.pyplot as plt
from data_process import train_dataloader, test_dataloader, valid_dataloader
from LeNet import evaluate_model, make_submit

def print_net_structure(net, X):
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)

def build_conv_block_in_dense(input_channels, output_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    )

def build_transition_block(input_channels, output_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

class DenseBlock(nn.Module):
    def __init__(self, num_conv, input_channels, output_channels):
        super().__init__()
        layer = []
        for i in range(num_conv):
            layer.append(build_conv_block_in_dense(
                input_channels + i * output_channels, output_channels))
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for block in self.net:
            Y = block(X)
            X = torch.cat((X, Y), dim=1)
        return X
    
block1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, padding=1)
)

current_channels, growth_rate = 64, 32
num_covs_all = [4, 4]
blocks = []

for i, num_covs in enumerate(num_covs_all):
    blocks.append(DenseBlock(num_covs, current_channels, growth_rate))
    current_channels += num_covs * growth_rate
    if i != len(num_covs_all) - 1:
        blocks.append(build_transition_block(current_channels, current_channels // 2))
        current_channels //= 2

net = nn.Sequential(
    block1,
    * blocks,
    nn.BatchNorm2d(current_channels),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(current_channels, 10)
)

print_net_structure(net, torch.rand(size=(1, 1, 28, 28)))

net = net.to("cuda")
epochs = 30
lr = 0.05
loss_fn = nn.CrossEntropyLoss()
losses = []
train_acc = []
valid_acc = []
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

for epoch in range(epochs):
    batch_loss = 0
    for idx, (x, y) in enumerate(train_dataloader):
        net.train()
        optimizer.zero_grad()
        x, y = x.to("cuda"), y.to("cuda")
        outputs = net(x)
        loss = loss_fn(outputs, y)
        batch_loss += loss.item()
        loss.backward()
        optimizer.step()
    losses.append(batch_loss / len(train_dataloader))
    train_acc.append(evaluate_model(net, train_dataloader))
    vac = evaluate_model(net, valid_dataloader)
    valid_acc.append(vac)
    print(f"第{epoch+1}训练:\nloss:{losses[-1]}\ntrain_acc:{train_acc[-1]}\nvalid_acc:{valid_acc[-1]}\n")
    if vac >=0.995 :
        print(f"测试集精度大于0.995，训练终止！")
        epochs = epoch+1
        break

plt.figure()
plt.plot(range(1, epochs + 1), losses, label="train_loss")
plt.plot(range(1, epochs + 1), train_acc, label="train_acc")
plt.plot(range(1, epochs + 1), valid_acc, label="valid_acc")
plt.legend(loc="upper right")
plt.show()

make_submit(net, test_dataloader, "DenseNet_submit.csv")
torch.save(net, r"kaggle/digit-recognizer/outputs/models/DenseNet.pth")