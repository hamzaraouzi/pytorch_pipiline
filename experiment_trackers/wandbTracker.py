import imp
from abstractTracker import AbstractTracker
import wandb

class WandBTracker(AbstractTracker):
    
    def __init__(self, project:str, tracking_conf:dict):
        self.tracking_conf = tracking_conf
        self.name = tracking_conf["name"]
        self.cridentials = tracking_conf["credentials"]
        self.project = project
    

    def init(self, config:dict):
        wandb.login(key=self.cridentials['key'])
        wandb.init(project=self.project, config=config)
    

    def log_metrics(self, metrics: dict):
        wandb.log(metrics)

    
    def __call__(self, metrics:dict):
        self.log_metrics(metrics)