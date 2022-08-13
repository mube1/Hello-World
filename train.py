import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from  tqdm import tqdm

class Trainer():
    def __init__(self, model,train_set,loss_criterion,optimizer,epoches,  printInterval=100,regularization=0):        
        self.accuracy = 0
        self.model=model
        self.optimizer=optimizer
        self.epoches=epoches
        self.loss_criterion=loss_criterion
        self.printInterval=printInterval
        self.train_set=train_set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.train_set
        self.regularization=regularization

    def training(self):
        regularization=self.regularization
        self.model.train()
        for epoch in range(self.epoches):
                
            for i, (images, labels) in enumerate(self.train_set): 
                
                
                images=images.reshape(-1,1,28,28).float()
                outputs = self.model(images)   
                
                labels = labels.to(self.device)        
                # Forward pass
       
                loss = self.loss_criterion(outputs, labels) 
#                 loss = loss + l2_lambda * l2_norm
        
        
                
                
          #Replaces pow(2.0) with abs() for L1 regularization

                l2_lambda = 0.001 
#             hyperparameters
                l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())

                if self.regularization!=0:
                    loss = loss + l2_lambda * l2_norm

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                     
            print (epoch, ' of ',self.epoches,' epoches', loss.item())
        
        
