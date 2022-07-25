


from abc import abstractmethod
from wandbTracker import WandBTracker

class AbstractTracker:


    @abstractmethod
    def init(self, config, cridentials):
        pass
    
    @abstractmethod
    def log_metrics(self,metrics:dict):
        pass
    
    @staticmethod
    def prepareTracker(name, project:str, tracking_conf:dict):
        if name == "wandb":
            return WandBTracker(project= project, tracking_conf= tracking_conf)
        
        raise f"{name} either not exist, or it's integration not yet implemented "
    
    def __call__(self, metrics:dict):
        self.log_metrics(metrics)
    