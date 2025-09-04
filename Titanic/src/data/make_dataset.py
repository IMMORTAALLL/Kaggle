import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


train_df = pd.read_csv(r"kaggle/Titanic/data/processed/train_processed.csv")

features = torch.tensor(train_df.drop(columns=['Survived']).values, dtype=torch.float32).to("cuda")
labels = torch.tensor(train_df['Survived'].values, dtype=torch.float32).to("cuda")

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.001, random_state=42)

class TitanicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, i):
        return self.features[i], self.labels[i]
    
train_dataset = TitanicDataset(train_features, train_labels)
test_dataset = TitanicDataset(test_features, test_labels)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=64)

submit_data = pd.read_csv(r"kaggle/Titanic/data/processed/test_processed.csv")
submit_data = torch.tensor(submit_data.drop(columns=['Survived']).values, dtype=torch.float32).to("cuda")