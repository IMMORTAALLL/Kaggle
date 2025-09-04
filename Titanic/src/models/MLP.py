import sys
sys.path.append("/workspace/kaggle/Titanic/src/data")
from make_dataset import train_dataloader, test_dataloader, submit_data

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self, x):
        return self.net(x)
        
model = MLP().to("cuda")
optimizer = torch.optim.AdamW(model.parameters())
epochs = 700
losses = []
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    all_batch_loss_in_one_epoch = 0
    for batch_idx, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(x).squeeze()
        loss = loss_fn(outputs, y)
        all_batch_loss_in_one_epoch += loss.item()
        loss.backward()
        optimizer.step()
    losses.append(all_batch_loss_in_one_epoch/len(train_dataloader))
    if (epoch + 1) % 100 == 0:
        print(f"Epoch{epoch + 1}:loss {losses[-1]}")

plt.figure()
plt.plot(range(1, epochs+1), losses)
plt.show()

with torch.no_grad():
    right = 0
    size = 0
    for batch_idx, (x, y) in enumerate(test_dataloader):
        outputs = model(x).squeeze()
        predicts = (outputs >= 0.5).int()
        right += (predicts == y).sum()
        size += x.size(0)
    submit_outputs = model(submit_data).squeeze()
    submit_predicts = (submit_outputs >= 0.5).int()
    id = (pd.read_csv(r"/workspace/kaggle/Titanic/data/raw/test.csv"))['PassengerId'].values
    submit_df = pd.DataFrame({"PassengerId":id, "Survived":submit_predicts.cpu()})
    submit_df.to_csv(r"/workspace/kaggle/Titanic/src/predict/mlp_submit2.csv",index=False)
    print(f"测试集预测准确率为:{right/size}")
    torch.save(model, r"/workspace/kaggle/Titanic/outputs/models/model2.pth")