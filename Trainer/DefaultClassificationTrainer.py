from typing import Optional
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from .AbtsractTrainer import AbstractTrainer
from ..experiment_trackers.abstractTracker import AbstractTracker

class DefaultClassificationTrainer(AbstractTrainer):
    
    def __init__(self, config_path: str):

        super(DefaultClassificationTrainer, self).__init__(config_path=config_path)
        
        self.criterion = self.define_criterion()
        

        #TODO: 1) We need to think a way for logging in terminal with logger,
        #and we need to track experiments with weights&biases

    def define_criterion(self):                                                                 
        if self.task =='binary_classification':
                                        
            return nn.BCELoss()                                                                             
        
        elif self.task == 'classification':                                                                                        
            return nn.CrossEntropyLoss()                                                            
    

    def define_optimizer(self, model):                                               
        # ##############################################################                        
        # args: optimizer name , it will be loaded from the conf file  #                        
        #                                                              #                            
        # returns: one of the optimizers available on pytorch          #                        
        ################################################################                        
                                                                                                                                                                         
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
                weight_decay=self.optimizer_prameters['weight_decay'])                                          

        if self.optimizer.parameters['name'] == 'Adagrad':
            return optim.Adagrad(model.parameters(), lr=self.optimizer_prameters['lr'], lr_decay=self.optimizer_prameters['lr_decay'], weight_decay=self.optimizer_prameters['weight_decay'])                                             

        if self.optimizer.parameters['name'] == 'Adadelta':
            return optim.Adadelta(model.parameters(), lr=self.optimizer_parameters['lr'], weight_decay=self.optimizer_prameters['weight_decay'])
    


    def log_metrics(self,exp_tracker:AbstractTracker,  model, y_train_true, y_train_pred, y_val_true, y_val_pred, train_loss, val_loss):
        
        #calculate  accuracies
        train_accuracy = accuracy_score(y_train_true.detach().cpu(), y_train_pred.detach().cpu() > 0.5) 
        val_accuracy =accuracy_score(y_val_true.detach().cpu(), y_val_pred.detach().cpu() > 0.5)
        #calculate  F1-scores
        train_f1 = f1_score(y_train_true.detach().cpu(), y_train_pred.detach().cpu() > 0.5)
        val_f1 = f1_score(y_val_true.detach().cpu(), y_val_pred.detach().cpu() > 0.5)

        #TODO  calculate and logging other metrics 
        metrics = {'train_accuracy': train_accuracy, 'val_accuracy':val_accuracy, 'train_accuracy': train_loss, 'val_accuracy':val_loss,
                'train_f1_score':train_f1, 'val_f1_score': val_f1}
        
        exp_tracker.log_metrics(metrics=metrics)
        return train_accuracy, val_accuracy



    def train(self,  model,
        train_loader: DataLoader, val_loader: DataLoader):
        #this is just an early version of the code that, will be parts like logging
        #saving and registring models ...
        
        model.to(device = self.device)
        exp_tracker = self.define_exp_tracker()
        exp_tracker.init()

        logging.info(f"{self.experiment_tracker['name']} is initialized successfully")
        #training loop
        best_accuracy = 0
        for epoch in range(self.num_epochs):
            logging.info(f"epoch {epoch} is starting...")
            train_loss, y_train_true, y_train_pred = model.one_train_epoch(train_loader = train_loader, 
                        criterion = self.criterion, optimizer=self.optimizer, device=self.device)
            
            val_loss, y_val_true, y_val_pred = model.one_val_epoch(val_loader = val_loader, 
                                                criterion = self.criterion, device=self.device)

            #TODO: logg all classification metrics 
            train_accuracy, val_accuracy = self.log_metrics(exp_tracker, model, y_train_true, y_train_pred, y_val_true, y_val_pred,
                train_loss, val_loss)
            #logging  metrics on the standard output  once the  epoch is completed
            logging.info(f"epoch {epoch} : train_loss = {train_loss}, val_loss = {val_loss}, train_accuracy = {train_accuracy}, \
                validation_accuracy = {val_accuracy}")


            no_improvement = 0
            if val_accuracy > best_accuracy:
                #TODO: probebly we will need a directory for artifacts
                checkpoint_path = model.name
                
                #logging a message that a checkpoint is saving
                logging.info("save checkpoint")
                model.save_checkpoint(model= model, optimizer=self.optimizer, file_name=checkpoint_path)
                best_accuracy = val_accuracy
                no_improvement = 0
            
            # early stopping
            elif val_accuracy <= best_accuracy and no_improvement < self.early_stopping:
                no_improvement +=1
                logging.warning(f"no improvement for {no_improvement}/{self.early_stopping} epochs")
            elif val_accuracy <= best_accuracy and no_improvement == self.early_stopping:
                #log a message that no improvement has been made for the {no_improvement} epochs
                logging.warning(f"the training is being stoped since there is no improvemnt for {no_improvement}/{self.early_stopping} epochs")
                break
            
            

    #the run function must be defined in parent classes as astract method (AbstractTrainer, Step)
    def run(self,  model:nn.Module,
        train_loader: DataLoader, val_loader: Optional[DataLoader]):
         
        self.optimizer = self.define_optimizer(model)

        
        if val_loader is  not None:
            self.train(model, train_loader, val_loader)


    def __call__(self, model:nn.Module,
        train_loader: DataLoader, val_loader: Optional[DataLoader]):
        
        self.run(model, train_loader, val_loader)