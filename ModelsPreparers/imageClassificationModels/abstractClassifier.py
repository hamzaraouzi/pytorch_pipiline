
from abc import abstractmethod
import torch
import torch.nn as nn
from tqdm import tqdm
class AbstractClassifier(nn.Module):

    #def __init__(self):
    #   super(self, AbstractClassifier).__init__()


    @abstractmethod
    def forward(self, x:torch.TensorType):
        pass
    
    @abstractmethod
    def prepareModel(model_name: str, num_classes:int):
        pass

    
    def one_val_epoch(self, val_loader, criterion, device):

        loop = tqdm(val_loader, leave=True)
        self.eval()
        val_loss = 0
        #accuracies = torch.zeros(4, device=device) #4 is the number of labels it will be changed in other projects 
        all_pred = []
        all_true = []
        with torch.no_grad():
            for _, (X, y) in enumerate(loop):
                X, y = X.to(device), y.to(device)
                y_pred = self(X)

                loss = criterion(y_pred, y)
                val_loss += loss.item()

                #calculate accuracy for each label
                #y_pred = y_pred > 0.5

                #accuracies += y_pred.eq(y).sum(dim=0)
                all_pred.append(y_pred)
                all_true.append(y)
            return val_loss, torch.cat(all_true), torch.cat(all_pred)




    def one_train_epoch(self,train_loader, criterion, optimizer, device):
            
        self.train()
        train_loss  = 0

        loop = tqdm(train_loader, leave=True)
       
        all_pred = []
        all_true = []
        for _,(X, y) in enumerate(loop):
            
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = self(X)
            
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            all_pred.append(y_pred)
            all_true.append(y)

        loop.set_postfix(train_loss = loss.item())
        return train_loss/len(train_loader.dataset), torch.cat(all_true), torch.cat(all_pred)