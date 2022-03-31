
from pyparsing import null_debug_action
import torch.nn as nn
import yaml
from .imageClassificationModels.mobileNet_v3 import MobileNetV3
from .imageClassificationModels.mobileNetV2 import MobileNetV2
from .imageClassificationModels.xception import Xception
from .imageClassificationModels.abstractClassifier import AbstractClassifier


class ClassificationModel: #we need to inherent from the class step
    def __init__(self, config_path):
        super(ClassificationModel, self).__init__()
        params2values = self.load_check_conf_file(config_path)

        self.model_name = params2values['name']
        self.task = params2values['task']
        self.pretrained = params2values['pretrained']
        self.num_classes = params2values['num_classes']

    def load_check_conf_file(self, config_path):
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)
        

        params2values = {}
        for d in conf_values['classificationModel']:
            for k, v in zip(d.keys(), d.values()):
                params2values[k] = v
    
        return params2values
    
    def prepareModel(self, model_name, num_classes, pretrained) -> AbstractClassifier:
        if model_name.split('-')[0] == 'mobileNetV3':
            return MobileNetV3.prepareModel(model_name, num_classes)
        
        if model_name.split('-')[0] == 'mobileNetV2':
            return MobileNetV2.prepareModel(model_name, num_classes)
        
        if model_name.split('-')[0] == 'Xception':
            return Xception.prepareModel(model_name, num_classes)
        

    

    def run(self):
        return self.prepareModel(self.model_name, self.num_classes, self.pretrained)
    
    def __call__(self):
        return self.run()