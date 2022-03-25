import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import torch.optim as optim
from sklearn.metrics import accuracy_score
from AbtsractTrainer import AbstractTrainer

class DefaultClassificationTrainer(AbstractTrainer):
    
    def __init__(self, config_path: str):

        super(DefaultClassificationTrainer, self).__init__(config_path=config_path)
        
        self.criterion = self.define_criterion()
        self.optimizer = self.define_optimizer() 

        #TODO: 1) We need to think a way for logging in terminal with logger,
        #and mean while thinking about tools like MLFlow and TensorBoard

    def define_criterion(self):                                                                 
        if self.model.out_dim == 1 or self.task =='multilabel_classification':                                                         #
            #TODO: don't forge about criterions parameters,                                    
            # we need to have the ability to set what values we want                               
            return nn.BCELoss()                                                                             
        
        elif self.task == 'classfication':                                                                                        #
            return nn.CrossEntropyLoss()                                                            
    

    def define_optimizer(self):                                               
        # ##############################################################                        
        # args: optimizer name , it will be loaded from the conf file  #                        
        #                                                              #                            
        # returns: one of the optimizers available on pytorch          #                        
        ################################################################                        
                                                                                                        
        #for now i will return just ADAM for tests                                                                  
        # TODO: later I have to add all the rest algorithms that are available on pytorch       
        return  optim.Adam(self.model.parameters())                                             
                                                      


    def train(self,  model:nn.Module,
        train_loader: DataLoader, val_loader: DataLoader):
        #this is just an early version of the code that, will be parts like logging
        #saving and registring models ...
        
        model.to(device = self.device)

        #training loop
        best_accurcy = 0
        for epoch in range(self.num_epochs):
            #TODO: logging an epoch is staring

            train_loss, y_train_true, y_train_pred = model.one_train_epoch(train_loader = train_loader, 
                        citerion = self.criterion, optimizer=self.optmizer)
            
            val_loss, y_val_true, y_val_pred = model.one_val_epoch(val_laoder = val_loader, 
                                                citerion = self.criterion, device=self.device)
        
            #TODO: logging accuracies and losses and that an epoch is completed
            val_accuracy =None # it will be calculated
            


            no_improvement = 0
            if val_accuracy > best_accuracy:
                #TODO: probebly we will need a directory for artifacts
                checkpoint_path = model.name
                
                #TODO: logging a message that a checkpoint is saving
                model.save_checkpoint(model= model, optimizer=self.optimizer, file_name=checkpoint_path)
                best_accuracy = val_accuracy
                no_improvement = 0
            
            # early stopping
            elif val_accuracy == best_accuracy and no_improvement < self.early_stopping:
                no_improvement +=1
            
            elif val_accuracy == best_accuracy and no_improvement == self.early_stopping:
                #TODO: logg a message that no improvement has been made for the {no_improvement} epochs
                break
            
            
    ###TODO: code training with k-fold Cross Validation