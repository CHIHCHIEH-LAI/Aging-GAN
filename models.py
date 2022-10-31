import torch
import torch.nn as nn
import numpy as np
import math
import os
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

DIM = 64

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        block = [
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.BatchNorm2d(in_features)
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_features):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_features+1, DIM, 2, 2),
            nn.BatchNorm2d(DIM),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(DIM, 2*DIM, 2, 2),
            nn.BatchNorm2d(2*DIM),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*DIM, 4*DIM, 2, 2),
            nn.BatchNorm2d(4*DIM),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(4*DIM, 8*DIM, 2, 2),
            nn.BatchNorm2d(8*DIM),
            nn.ReLU()
        )
        
        model = [ResidualBlock(8*DIM) for _ in range(3)]
        self.layer5 = nn.Sequential(*model)

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(8*DIM, 4*DIM, 2, 2),
            nn.BatchNorm2d(4*DIM),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 2, 2),
            nn.BatchNorm2d(2*DIM),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 2, 2),
            nn.BatchNorm2d(DIM),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.ConvTranspose2d(DIM, in_features+1, 2, 2),
            nn.Tanh()
        )

    def forward(self, img, age):

        x = torch.cat((img, age), 1)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.layer5(x4) + x4
        
        x6 = self.layer6(x5) + x3
        x7 = self.layer7(x6) + x2
        x8 = self.layer8(x7) + x1
        x9 = self.layer9(x8)

        img = x9[:,0:3,:,:]

        return img

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_features+1, DIM, 3, 2, padding=1),
            nn.BatchNorm2d(DIM),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(DIM, 2*DIM, 3, 2, padding=1),
            nn.BatchNorm2d(2*DIM),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*DIM, 4*DIM, 3, 2, padding=1),
            nn.BatchNorm2d(4*DIM),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(4*DIM, 8*DIM, 3, 2, padding=1),
            nn.BatchNorm2d(8*DIM),
            nn.LeakyReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(8*DIM, 16*DIM, 3, 2, padding=1),
            nn.BatchNorm2d(16*DIM),
            nn.LeakyReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(16*DIM*4*4, 2)
        )
    
    def forward(self, img, age):

        x = torch.cat((img, age), 1)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x5 = x5.view(x5.shape[0], -1)
        validity = self.layer6(x5)

        return validity