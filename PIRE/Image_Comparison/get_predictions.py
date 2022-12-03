import torch
import numpy as np
import pandas as pd

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

import time, os, copy, glob


def get_all_files_in_dir(dirName:str) -> list:
    return glob.glob(dirName)

def get_file_names(dirName:str) -> list:
    files = get_all_files_in_dir(dirName)
    return list(map(os.path.basename, files))

def get_image_file(filepath:str) -> Image:
    with Image.open(filepath) as img:
        img.load()
    return img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DF_COLUMNS = ['Image', 'GT_Pred', 'Adv_Pred', 'Same']
INDEX = get_file_names('./gt_queries/*.jpg')

class ImageDataset(Dataset):
    def __init__(self, dirName:str, transform = None) -> None:
        super().__init__()
        filenames = get_all_files_in_dir(dirName)
        self.image_files = list(map(get_image_file, filenames))
        self.transform = transform.transforms()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx) :
        transformed = self.transform(self.image_files[idx])
        return torch.unsqueeze(transformed, 0).to(device)
        

def get_model(model_type) -> torch.nn.Module:
    model = model_type(pretrained=True).to(device)
    model.eval()
    return model

def predict_image_class(model: torch.nn.Module, image: torch.Tensor) -> int:
    out = model(image)
    _, index = torch.max(out, 1)
    return index[0].item()

def create_preds_df() -> pd.DataFrame:
    df = pd.DataFrame(columns=DF_COLUMNS)
    df['Image'] = INDEX
    return df.fillna(0)




def get_gt_labels(df: pd.DataFrame, model_type = models.resnet101, transform = models.ResNet101_Weights.IMAGENET1K_V2) -> pd.DataFrame:
    dataset = ImageDataset('./gt_queries/*.jpg', transform)
    model = get_model(model_type)
    for i in range(len(dataset)):
        pred = predict_image_class(model = model, image = dataset[i])
        df.loc[i, 'GT_Pred'] = pred
    
    return df

def get_adv_labels(df: pd.DataFrame, model_type = models.resnet101, transform = models.ResNet101_Weights.IMAGENET1K_V2) -> pd.DataFrame:
    dataset = ImageDataset('./adv_queries/*.jpg', transform)
    model = get_model(model_type)
    for i in range(len(dataset)):
        pred = predict_image_class(model = model, image = dataset[i])
        df.loc[i, 'Adv_Pred'] = pred
    
    return df

def mark_columns_same(df: pd.DataFrame) -> pd.DataFrame:
    same = (df['GT_Pred'] == df['Adv_Pred']).astype(int)
    df['Same'] = same
    return df

def get_success_rate(df: pd.DataFrame):
    success_rate = len(df[df['Same'] == 0])/len(df['Same'])
    print('Success Rate: %.2f' %(success_rate*100))

def main():
    df = create_preds_df()
    df = get_gt_labels(df = df)
    df = get_adv_labels(df = df)
    df = mark_columns_same(df)
    df.to_csv('preds.csv', index = True)
    get_success_rate(df)
    

if __name__ == '__main__':
    main()