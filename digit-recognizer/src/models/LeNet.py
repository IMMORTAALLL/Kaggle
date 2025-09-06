import sys
sys.path.append(r"/workspace/kaggle/digit-recognizer/src/data")

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from data_process import train_dataloader, valid_dataloader, test_dataloader, train_dataloader_max

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),nn.Sigmoid(),nn.Dropout(0.2),
            nn.Linear(120,84),nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        return self.net(x)
    
class LeNet_MR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),nn.BatchNorm2d(6),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),nn.BatchNorm2d(16),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),nn.ReLU(),nn.Dropout(0.1),
            nn.Linear(120,84),nn.ReLU(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        return self.net(x)
    
#model = LeNet().to("cuda")
model = LeNet_MR().to("cuda")
#epochs = 50
epochs = 50
lr = 0.1
loss_fn = nn.CrossEntropyLoss()
losses = []
train_acc = []
valid_acc = []
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            x, y = x.to("cuda"), y.to("cuda")
            outputs = model(x)
            predicts = torch.argmax(outputs, dim=1)
            all_preds.extend(predicts.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

def make_submit(model, dataloader, file_name):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to("cuda")
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().squeeze())
    ImageId = torch.arange(1,len(all_preds) + 1)
    df = pd.DataFrame({"ImageId":ImageId, "Label":all_preds})
    df.to_csv(os.path.join("kaggle/digit-recognizer/src/predict", file_name), index=False)




if __name__ == "__main__":

    for epoch in range(epochs):
        batch_loss = 0
        for idx, (x, y) in enumerate(train_dataloader_max):
            model.train()
            optimizer.zero_grad()
            x, y = x.to("cuda"), y.to("cuda")
            outputs = model(x)
            loss = loss_fn(outputs, y)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(batch_loss / len(train_dataloader_max))
        train_acc.append(evaluate_model(model, train_dataloader_max))
        valid_acc.append(evaluate_model(model, valid_dataloader))
        print(f"第{epoch+1}训练:\nloss:{losses[-1]}\ntrain_acc:{train_acc[-1]}\nvalid_acc:{valid_acc[-1]}\n")

    plt.figure()
    plt.plot(range(1, epochs + 1), losses, label="train_loss")
    plt.plot(range(1, epochs + 1), train_acc, label="train_acc")
    plt.plot(range(1, epochs + 1), valid_acc, label="valid_acc")
    plt.legend(loc="upper right")
    plt.show()

    make_submit(model, test_dataloader, "LeNet_MR_PRO_MAX_submit.csv")
    torch.save(model, r"kaggle/digit-recognizer/outputs/models/LeNet_MR_PRO_MAX.pth")