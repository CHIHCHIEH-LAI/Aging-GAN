import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import math
import os
import random
from PIL import Image
import pdb


def split_data():
    os.makedirs("save/training_data", exist_ok=True)
    os.makedirs("save/training_data/train", exist_ok=True)
    os.makedirs("save/training_data/test", exist_ok=True)
    
    skip = True
    for (path, dirs, files) in sorted(os.walk('./data')):
        if skip==True:
            skip=False
        else:
            for filename in (files):
                file_id = (int)(filename[0:5])
                img = Image.open(path+"/"+filename).convert('RGB')
                if file_id < 65000:
                    img.save("save/training_data/train/"+filename) 
                else:
                    img.save("save/training_data/test/"+filename)   

class customDataset(Dataset):

    def __init__(self, transform, mode, txt_file):
        self.transform = transform
        self.files_path = "save/training_data/" + mode
        self.files = sorted(os.listdir(self.files_path))
        self.labels = np.loadtxt(txt_file, dtype=int, usecols=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        idx2 = random.randint(0, len(self.files) - 1)

        imgA = self.transform(Image.open(os.path.join(self.files_path, self.files[idx])).convert('RGB'))
        imgB = self.transform(Image.open(os.path.join(self.files_path, self.files[idx2])).convert('RGB'))
        
        lblA = self.labels[idx]//10
        lblB = self.labels[idx2]//10

        if lblA > lblB:
            return imgA, imgB, lblA*0.001, lblB*0.001
        else:
            return imgB, imgA, lblB*0.001, lblA*0.001

class customDataset1(Dataset):

    def __init__(self, transform):
        self.transform = transform
        self.files_path = "save/training_data/test"
        self.files = sorted(os.listdir(self.files_path))
        self.labels = np.loadtxt('test_label.txt', dtype=int, usecols=1)
        self.desired_age_labels = np.loadtxt('test_desired_age.txt', dtype=int, usecols=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img = self.transform(Image.open(os.path.join(self.files_path, self.files[idx])).convert('RGB'))

        return img, (self.labels[idx]//10)*0.001, (self.desired_age_labels[idx]//10)*0.001