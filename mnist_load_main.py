from torchvision import datasets, transforms
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class train_data(Dataset):
    def __init__(self):
        
        self.data = pd.read_csv('./mnist_csv/mnist_train.csv').iloc[:][1:]
#         np.array(train)
#         np.array(train.iloc[:][1:])
        self.x=np.array(self.data.iloc[:,1:])
        self.y= np.array(self.data.iloc[:,0])
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class test_data(Dataset):
    def __init__(self):
        self.data = pd.read_csv('./mnist_csv/mnist_test.csv').iloc[:][1:]
        self.x=torch.tensor(np.array(self.data.iloc[:,1:]))
        self.y= torch.tensor(np.array(self.data.iloc[:,0]))
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    



def get_x_and_y():
    train_loader = torch.utils.data.DataLoader(dataset=train_data(), batch_size=32)
    test_loader = torch.utils.data.DataLoader(dataset=test_data(), batch_size=32,shuffle=False)
    return train_loader, test_loader
    
