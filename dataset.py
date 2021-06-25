import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
input_size = (64,32)


def one_hot_encoding(labels,num_classes):
	return torch.eye(num_classes)[labels.data.cpu()]


class MyDataset(Dataset):
    def __init__(self, df,data_transform):
        self.df=df
        self.data_transform=data_transform
    def __len__(self):
        return self.df.shape[0]
    
    def get_bbox(self, idx,shpe):
        bboxes=[]
        for i in range(1,6):
            bbox = list(eval(self.df.loc[idx,'bbox%s'%i]))
            bbox[0] = float(bbox[0]/shpe[0])
            bbox[1] = float(bbox[1]/shpe[1])
            bbox[2] = float((bbox[0]+bbox[2])/shpe[0])
            bbox[3] = float((bbox[1]+bbox[3])/shpe[1])
            bboxes.append(bbox)
        return torch.tensor(bboxes, dtype=torch.float32)
    
    def get_class(self, idx):
        clss=[]
        for i in range(1,6):
            cls = self.df.loc[idx,'num%s'%i]
            clss.append(cls)
        return one_hot_encoding(torch.tensor(clss),11)

    def get_img(self,idx):
    	im = Image.open(self.df.loc[idx,'filename']).convert('RGB')
    	w,h=im.size
    	return Image.open(self.df.loc[idx,'filename']).convert('RGB'),(w,h)
        

    def __getitem__(self, idx):
        img,shpe=self.get_img(idx)
        return self.data_transform(img),[self.get_bbox(idx,shpe),self.get_class(idx)]

