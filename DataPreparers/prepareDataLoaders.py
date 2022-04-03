
from cv2 import split
import yaml
import pandas as pd
from supervisedImageClassificationDataset import SupervisedImageClassicationDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class PrepareDataLoader:

    def __init__(self, config_path):

        parameters = self.load_check_conf_file(config_path)

        self.task     = parameters['task']    
        self.csv_file = parameters['csv_file']
        self.img_dir  = parameters['img_dir']   
        self.target_column = parameters['target_column']
        self.split = parameters['split']
        self.ratios = parameters['ratios'] # {train: ,val: ,test: } or {train: ,test: } 
        ###TODO we need to add splits ratios
        ######################################
        self.sampling = parameters['sampling']
        self.batch_size = parameters['batch_size']
        
         

    def prepareSupervisedImageClassicationDataset(self, df):
        return SupervisedImageClassicationDataset(img_dir=self.img_dir, df=df, target_column=self.target_column,
        sampling=self.sampling, transform=None)

    def __call__(self):
        #TODO handel ratios and diffrents splits
        df = pd.read_csv(self.csv_file)
        
        train_df, test_df = train_test_split(df, test_size=self.ratios['test'])
        if self.split == 'train-test':
            
            
            train_ds = SupervisedImageClassicationDataset(img_dir = self.img_dir, df=train_df, 
            target_column=self.target_column, sampling=None, transform=None)

            test_ds = SupervisedImageClassicationDataset(img_dir = self.img_dir, df=test_df, 
            target_column=self.target_column, sampling=None, transform=None)
            
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
            return train_loader, test_loader

        if self.split == 'train-val-test':

            val_df , test_df = train_test_split(test_df, test_size=self.ratios['test']/(self.ratios['test'] + self.ratios['val'])) 

            train_ds = SupervisedImageClassicationDataset(img_dir = self.img_dir, df=train_df, 
            target_column=self.target_column, sampling=None, transform=None)

            test_ds = SupervisedImageClassicationDataset(img_dir = self.img_dir, df=test_df, 
            target_column=self.target_column, sampling=None, transform=None)
            

            val_ds = SupervisedImageClassicationDataset(img_dir = self.img_dir, df=val_df, 
            target_column=self.target_column, sampling=None, transform=None)

            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

            return train_loader, val_loader, test_loader 


    def load_check_conf_file(self, config_path):
        with open(config_path) as file:
            conf_values = yaml.load(file, Loader=yaml.FullLoader)
        

        params2values = {}
        for d in conf_values['Dataset']:
            for k, v in zip(d.keys(), d.values()):
                params2values[k] = v
    
        return params2values