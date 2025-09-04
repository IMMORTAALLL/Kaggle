import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r"/data/kaggle/titanic/train.csv")
df1 = pd.read_csv(r"/data/kaggle/titanic/test.csv")
df.drop(columns=['Cabin','Name','Ticket','Fare','Embarked','PassengerId'],inplace=True)
df.dropna(inplace=True)
test_id = df1['PassengerId'].values
df1.drop(columns=['Cabin','Name','Ticket','Fare','Embarked','PassengerId'],inplace=True)
df1.fillna(df1['Age'].mean(),inplace=True)
print(df1.info())
df1["Sex"] = df1["Sex"].map({'male':0,'female':1})
df["Sex"] = df["Sex"].map({'male':0,'female':1})
#print(df.head())

#print(((df['Age'] >= 40)).sum() / df.shape[0])

def group_age(age):
    if age < 18:
        return 0
    elif age < 30:
        return 1
    elif age < 40:
        return 2
    else :
        return 3
    
df['Age'] = df['Age'].apply(group_age)
df1['Age'] = df1['Age'].apply(group_age)
#print(df.head())

labels = df['Survived'].values
features = df.drop(columns=['Survived']).values





class TitanicDataset(Dataset):
    def __init__(self,features,labels):
        self.features = torch.tensor(features,dtype=torch.float32).to('cuda')
        self.labels = torch.tensor(labels,dtype=torch.float32).to('cuda')
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx]
    
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42  
)

scaler = StandardScaler()
scaler.fit(train_features)
train_features_standardized = scaler.transform(train_features)
test_features_standardized = scaler.transform(test_features)
df1_test_features_standardized = scaler.transform(df1.values)

train_dataset = TitanicDataset(train_features_standardized, train_labels)
test_dataset = TitanicDataset(test_features_standardized, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print("训练集批次示例：")
for batch_idx, (X, y) in enumerate(train_loader):
    print(f"批次 {batch_idx+1}：")
    print(f"  特征形状：{X.shape}")  # 应为(batch_size, 5)，5个特征
    print(f"  特征数据：\n{X}")
    print(f"  标签数据：{y}\n")

class DNN(nn.Module):
    def __init__(self):
        super(DNN,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5,16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self,x):
        return self.model(x)
    
net = DNN().to('cuda')

loss_fn = nn.BCEWithLogitsLoss()
epoch_losses = []
epochs = 2000
lr = 0.01
#optimizer = torch.optim.SGD(net.parameters(),lr=lr,weight_decay=1e-4, momentum=0.99)
optimizer = torch.optim.Adam(net.parameters())
#optimizer = torch.optim.RAdam(net.parameters())

for epoch in range(epochs):
    net.train()
    batch_loss = 0
    for batch_idx,(X,y) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(X).squeeze()
        loss = loss_fn(outputs,y)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
    epoch_losses.append(batch_loss/len(train_loader))
    if (epoch+1) % 100 == 0:
        print(f"[{epoch+1}/{epochs}]loss:{epoch_losses[-1]}")

plt.figure()
plt.plot(range(epochs), epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
print(df1.head())
with torch.no_grad():
    net.eval()
    right = 0
    right1 = 0
    for batch_idx,(x,y) in enumerate(test_loader):
        outputs = net(x).squeeze()
        predicted = (outputs >= 0.5).float()
        right += (predicted == y).sum().item()
        predicted1 = (x[:,1] == 0).float()
        right1 += (predicted1 == y).sum().item()
    accuracy = right / len(test_dataset)
    print(accuracy)
    print(right1/len(test_dataset))
    df1_test_features_tensor = torch.tensor(df1_test_features_standardized,dtype=torch.float32).to("cuda")
    df1_test_outputs = net(df1_test_features_tensor).squeeze()
    sub_df = pd.DataFrame({"PassengerId":test_id, "Survived":(df1_test_outputs >= 0.5).int().cpu()})
    print(sub_df.head())
    sub_df.to_csv(r"submmit.csv", index=False)

print(df.head())