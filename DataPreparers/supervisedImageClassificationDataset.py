import torch
from  torch.utils.data import Dataset
import os
from  PIL import Image 
from torchvision import transforms
import pandas as pd


class SupervisedImageClassicationDataset(Dataset):

    def __init__(self, img_dir:str, df:pd.DataFrame, target_column:str, sampling:str
        ,transform:transforms):
        

        self.df = df
        self.target_column = target_column
        
        self.img_dir = img_dir
        self.transform = transform
        
    

        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        y = self.df.loc[idx, self.target_column]
        image_name = self.df.loc[idx, image_name]
        
        img_path = os.path.join(self.img_dir, image_name)
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), torch.tensor(y)        