from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import torch.optim as optim
from sklearn.metrics import accuracy_score

class DefaultClassificationTrainer:
    
    def __init__(self, config_file: str, model:nn.Module, device:str, \
        train_loader: DataLoader,
        num_epochs:int, val_loader: DataLoader):

        # assert  early_stopping > 2 and num_epochs > early_stopping, 'please choose a proper earlystoping value '
        # earlystopping will be loaded from the yaml file, and then I get back the assert test
        
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = self.define_criterion()
        self.optimizer = self.define_optimizer() 

        #TODO: 1) We need to think a way for logging in terminal with logger,
        #and mean while thinking about tools like MLFlow and TensorBoard



#HINT: if we will be creating an AbstractTrainer, probably it's reasonable to define this functions there.

#################################################################################################  
    def define_optimizer(self, optim_name):                                                     #    
        # ##############################################################                        #
        # args: optimizer name , it will be loaded from the conf file  #                        #
        #                                                              #                        #    
        # returns: one of the optimizers available on pytorch          #                        #
        ################################################################                        #
                                                                                                #        
        #for now i will return just ADAM for tests                                              #                    
        # TODO: later I have to add all the rest algorithms that are available on pytorch       #
        return  optim.Adam(self.model.parameters())                                             #
                                                                                                #    
                                                                                                #
    def define_criterion(self):                                                                 #
        if self.model.num_classes == 1:                                                         #
            #TODO: don't forge about criterions parameters,                                     #
            # we need to have the ability to set what values we want                            #    
            return nn.BCELoss()                                                                 #            
                                                                                                #
        return nn.CrossEntropyLoss()                                                            #
                                                                                                #
#################################################################################################
        


    def train(self):
        #this is just an early version of the code that, will be parts like logging
        #saving and registring models ...
        
        self.model.to(device = self.device)

        #training loop
        best_accurcy = 0
        for epoch in range(self.num_epochs):
            #TODO: logging an epoch is staring

            train_loss, y_train_true, y_train_pred = self.model.one_train_epoch(train_loader = self.train_loader, 
                        citerion = self.criterion, optimizer=self.optmizer)
            
            val_loss, y_val_true, y_val_pred = self.model.one_val_epoch(val_laoder=self.val_loader, 
                                                citerion = self.criterion, device=self.device)
        
            #TODO: calculate tarining / validation metrics and logging it
            val_accuracy =None # it will be calculated
            
            no_improvement = 0
            if val_accuracy > best_accuracy:
                #probebly we will need a directory for artifacts
                checkpoint_path = self.model.name
                
                self.model.save_checkpoint(model= self.model, optimizer=self.optimizer, file_name=checkpoint_path)
                best_accuracy = val_accuracy
                no_improvement = 0
            
            
            elif val_accuracy == best_accuracy and no_improvement < early_stopping:
                no_improvement +=1
            
            elif val_accuracy == best_accuracy and no_improvement == early_stopping:
                break
            

            #TODO: implement early stopping test
            
    ###TODO: code training with k-fold Cross Validation