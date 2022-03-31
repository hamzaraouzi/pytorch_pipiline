
from cv2 import split
import yaml
import pandas as pd
from supervisedImageClassificationDataset import SupervisedImageClassicationDataset




class PrepareDataLoader:

    def __init__(self, config_path):

        parameters = self.load_check_conf_file(config_path)

        self.task     = parameters['task']    
        self.csv_file = parameters['csv_file']
        self.img_dir  = parameters['img_dir']   
        self.target_column = parameters['target_column']
        self.split = parameters['split']
        ###TODO we need to add splits ratios
        ######################################
        self.sampling = parameters['sampling']

        
         

    def prepareSupervisedImageClassicationDataset(self, df):
        return SupervisedImageClassicationDataset(img_dir=self.img_dir, df=df, target_column=self.target_column,
        sampling=self.sampling, transform=None)

    def __call__(self):
        #TODO handel ratios and diffrents splits
        if self.split == 'train-test':
            pass
        
        if self.split == 'train-val-test':
            pass


    def load_check_conf_file(self, config_path):
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)
        

        params2values = {}
        for d in conf_values['Dataset']:
            for k, v in zip(d.keys(), d.values()):
                params2values[k] = v
    
        return params2values