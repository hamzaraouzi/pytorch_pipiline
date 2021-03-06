from typing import Optional
from abc import abstractmethod
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from ..experiment_trackers.abstractTracker import AbstractTracker
class AbstractTrainer:
    def __init__(self, config_path) -> None:
        #self.logger
        
        params2values, self.optimizer_prameters = self.load_check_conf_file(config_path)

        self.task = params2values['task']
        self.device = params2values['device']
        self.num_epochs = params2values['num_epochs']
        self.early_stopping = params2values['earlystoping_after']
        self.project = params2values['project']
        self.experiment_tracker = params2values['experiment_tracker']
        


    def load_check_conf_file(self, config_path):        
        #I'm just loading for now
        # TODO: we need to check paramters
        
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)

        params2values = dict()
        
        params2values = {}
        optimizer_parameters = {}

        for d in conf_values['training']:
            for k, v in zip(d.keys(), d.values()):
                if k!='optimizer':
                    params2values[k] = v
                
                else:
                    for dd in v:
                        for kk, vv in zip(dd.keys(), dd.values()):
                            if kk!='optimizer':
                                optimizer_parameters[kk] = vv
        

        return params2values, optimizer_parameters



    def define_exp_tracker(self)->AbstractTracker:
        return AbstractTracker.prepareTracker(project= self.project, tracking_conf=self.experiment_tracker)
    

    @abstractmethod
    def define_criterion(self):
        pass

    @abstractmethod
    def define_optimizer(self, optimizer_parameters):
        pass                                              
                                                                                                
    
    @abstractmethod
    def train(self,  model:nn.Module,
        train_loader: DataLoader, val_loader: DataLoader):
        pass
    
    @abstractmethod
    def run(self,  model:nn.Module,
        train_loader: DataLoader, val_loader: Optional[DataLoader]):
        pass
    

    @abstractmethod
    def log_metrics(self, exp_tracker:AbstractTracker, **kwrags):
        pass
    