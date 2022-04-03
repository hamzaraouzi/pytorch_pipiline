from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import torch.optim as optim
from sklearn.metrics import accuracy_score
from .AbtsractTrainer import AbstractTrainer

class DefaultClassificationTrainer(AbstractTrainer):
    
    def __init__(self, config_path: str):

        super(DefaultClassificationTrainer, self).__init__(config_path=config_path)
        
        self.criterion = self.define_criterion()
        

        #TODO: 1) We need to think a way for logging in terminal with logger,
        #and mean while thinking about tools like MLFlow and TensorBoard

    def define_criterion(self):                                                                 
        if self.task =='binary_classification':
                                        
            return nn.BCELoss()                                                                             
        
        elif self.task == 'classification':                                                                                        #
            return nn.CrossEntropyLoss()                                                            
    

    def define_optimizer(self, model):                                               
        # ##############################################################                        
        # args: optimizer name , it will be loaded from the conf file  #                        
        #                                                              #                            
        # returns: one of the optimizers available on pytorch          #                        
        ################################################################                        
                                                                                                        
        #for now i will return just ADAM for tests                                                                  
        # TODO: later I have to add all the rest algorithms that are available on pytorch       
        if self.optimizer_prameters['name'] == 'Adam':
            return  optim.Adam(model.parameters(), lr=self.optimizer_prameters['lr'],
             betas=self.optimizer_prameters['betas'], weight_decay=self.optimizer_prameters['weight_decay'])
        
        if self.optimizer_prameters['name'] =='SGD':
            return optim.SGD(model.parameters(), lr=self.optimizer_prameters['lr'], 
                    momentum=self.optimizer_prameters['momentum'], dampening=self.optimizer_prameters['dampening'],
                    nestrove=self.optimizer_prameters['nestrove'])

        if self.optimizer.parameters['name'] == 'RMSprop':
            return optim.RMSprop(model.parameters(), lr=self.optimizer_prameters['lr'],
                momentum=self.optimizer_prameters['momentum'], alpha=self.optimizer_prameters['alpha'],
                weight_decay=self.optimizer_prameters['weight_decay'] )                                          
                                                      
    #def log_metrics():




    def train(self,  model,
        train_loader: DataLoader, val_loader: DataLoader):
        #this is just an early version of the code that, will be parts like logging
        #saving and registring models ...
        
        model.to(device = self.device)

        #training loop
        best_accurcy = 0
        for epoch in range(self.num_epochs):
            #TODO: logging an epoch is staring

            train_loss, y_train_true, y_train_pred = model.one_train_epoch(train_loader = train_loader, 
                        criterion = self.criterion, optimizer=self.optimizer, device=self.device)
            
            val_loss, y_val_true, y_val_pred = model.one_val_epoch(val_loader = val_loader, 
                                                criterion = self.criterion, device=self.device)
        
            
            val_accuracy =accuracy_score(y_val_true.detach().cpu(), y_val_pred.detach().cpu() > 0.5) 
            #TODO: logging accuracies and losses and that an epoch is completed


            no_improvement = 0
            if val_accuracy > best_accuracy:
                #TODO: probebly we will need a directory for artifacts
                checkpoint_path = model.name
                
                #TODO: logging a message that a checkpoint is saving
                model.save_checkpoint(model= model, optimizer=self.optimizer, file_name=checkpoint_path)
                best_accuracy = val_accuracy
                no_improvement = 0
                print(f'validation accuracy is : {val_accuracy}')
            
            # early stopping
            elif val_accuracy == best_accuracy and no_improvement < self.early_stopping:
                no_improvement +=1
            
            elif val_accuracy == best_accuracy and no_improvement == self.early_stopping:
                #TODO: logg a message that no improvement has been made for the {no_improvement} epochs
                break
            
            
    ###TODO: code training with k-fold Cross Validation


    #the run function must be defined in parent classes as astract method (AbstractTrainer, Step)
    def run(self,  model:nn.Module,
        train_loader: DataLoader, val_loader: Optional[DataLoader], k=Optional[int]):
         
        self.optimizer = self.define_optimizer(model)

        if not self.kfold:
            if val_loader is  not None:
                self.train(model, train_loader, val_loader)

            #else:
                #logging warning or Error that the user must either use Train/val/test split or kfoldCrossvalidation   
        #else:
            # training kfold cross validation 

    def __call__(self, model:nn.Module,
        train_loader: DataLoader, val_loader: Optional[DataLoader], k=Optional[int]):
        
        self.run(model, train_loader, val_loader, k)