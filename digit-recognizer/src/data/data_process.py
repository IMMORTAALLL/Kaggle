import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


train_df = pd.read_csv(r"kaggle/digit-recognizer/data/raw/train.csv")
test_df = pd.read_csv(r"kaggle/digit-recognizer/data/raw/test.csv")
all_df = pd.concat([train_df, test_df], ignore_index=True)
images = torch.tensor(all_df.drop(columns=['label']).values, dtype=torch.float32).reshape(-1, 28, 28)
features = (images[:((~(pd.isnull(all_df['label']))).sum()), :] / 255.0).unsqueeze(1)
labels = torch.tensor(train_df['label'].values, dtype=torch.float32)
test_features = (images[features.shape[0]:,:] / 255.0).unsqueeze(1)

train_features, valid_features, train_labels, valid_labels = train_test_split(
    features, labels,
    random_state=42,
    test_size=0.2,
    stratify=labels
)

class MNISTDataset(Dataset):
    def __init__(self, features, labels, transform):
        super().__init__()
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return self.transform(feature), label
    
class MNISTTestDataset(Dataset):
    def __init__(self, features,transform):
        super().__init__()
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return self.transform(feature)

transform = transforms.Compose([
    transforms.Normalize(mean=train_features.mean(),std=train_features.std())
])



train_dataset = MNISTDataset(train_features, train_labels, transform)
valid_dataset = MNISTDataset(valid_features, valid_labels, transform)
test_dataset = MNISTTestDataset(test_features, transform)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size = 256)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size = 128)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size = 256)


if __name__ == "__main__":
    #print(train_df.info())
    print(features.shape)
    print(len(features))
    print(train_features.mean(), train_features.std())
    print(len(train_dataset), len(valid_dataset))
    for idx, x in enumerate(test_dataloader):
        print(f"{idx}:{x.shape}")
    