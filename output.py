import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import os
import argparse
from PIL import Image
import pdb

from models import Generator
from dataset import split_data, customDataset1

# Setting
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=int, default=50, help="choose a model from save/trained_model")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--split_data", type=bool, default=False, help="split the data into training set and testing set")
opt = parser.parse_args()

if opt.split_data:
    split_data()

model_id = str(opt.model_id)

# Create directory
os.makedirs("save/gen_imgs", exist_ok=True)
save_path = "save/gen_imgs/"+str(model_id)
os.makedirs(save_path, exist_ok=True)

cuda = torch.cuda.is_available()

# Initialize generator and discriminator
G_A2B = Generator(opt.channels)
G_B2A = Generator(opt.channels)

# Load generator models
load_G_A2B = torch.load("save/trained_model/"+str(model_id)+"/G_A2B.pth")
G_A2B.load_state_dict(load_G_A2B)
load_G_B2A = torch.load("save/trained_model/"+str(model_id)+"/G_B2A.pth")
G_B2A.load_state_dict(load_G_B2A)

if cuda:
    G_A2B.cuda()
    G_B2A.cuda()

G_A2B.eval()
G_B2A.eval()

# Configure data loader
test_loader = DataLoader(
    customDataset1(
        transform = transforms.Compose([
            transforms.Resize(opt.img_size), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    ),
    batch_size=opt.batch_size,
    shuffle=False
)

print("number of images: " + str(len(test_loader)))

FTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

with torch.no_grad():

    for i, (image, lbl, desired_age_lbl) in enumerate(test_loader):

        print(i, "\t\t\r", end="")

        img = Variable(image.type(FTensor))

        age = Variable(desired_age_lbl.type(FTensor))
        age = age.unsqueeze(1)
        age = age.repeat(1, opt.img_size**2)
        age = age.view(opt.batch_size, 1, opt.img_size, opt.img_size)

        if lbl > desired_age_lbl:
            gen = G_A2B(img, age)
        else:
            gen = G_B2A(img, age)

        age = Variable(lbl.type(FTensor))
        age = age.unsqueeze(1)
        age = age.repeat(1, opt.img_size**2)
        age = age.view(opt.batch_size, 1, opt.img_size, opt.img_size)

        if lbl > desired_age_lbl:
            reconstruct = G_B2A(gen, age)
        else:
            reconstruct = G_A2B(gen, age)

        save_image(gen.data[0], save_path+"/"+str(65000+i)+"_aged.png", nrow=1, normalize=True)
        save_image(reconstruct.data[0], save_path+"/"+str(65000+i)+"_rec.png", nrow=1, normalize=True)