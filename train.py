import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import math
import os
import itertools
import argparse
import pdb

from visualize import *
from dataset import *
from models import *

os.makedirs("save", exist_ok=True)
os.makedirs("save/trained_model", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5, help="interval betwen epochs")
parser.add_argument("--split_data", type=bool, default=False, help="split the data into training set and testing set")
opt = parser.parse_args()
print(opt)

if opt.split_data:
	split_data()

cuda = torch.cuda.is_available()

# Initialize generator and discriminator
G_A2B = Generator(opt.channels)
G_B2A = Generator(opt.channels)
D_A = Discriminator(opt.channels)
D_B = Discriminator(opt.channels)

G_params = count_parameters(G_A2B)
D_params = count_parameters(D_A)
print("G_params:"+str(G_params))
print("D_params:"+str(D_params))

if cuda:
	G_A2B.cuda()
	G_B2A.cuda()
	D_A.cuda()
	D_B.cuda()

# Loss function
criterion_GAN = nn.CrossEntropyLoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

if cuda:
	criterion_GAN.cuda()
	criterion_cycle.cuda()
	criterion_identity.cuda()

# Optimizers
optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=opt.lr)
optimizer_D_A = optim.Adam(D_A.parameters(), lr=opt.lr)
optimizer_D_B = optim.Adam(D_B.parameters(), lr=opt.lr)

# # Learning rate update schedulers
# lr_scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=20, gamma=0.05)
# lr_scheduler_D_A = torch.optim.lr_scheduler.StepLR(optimizer_D_A, step_size=20, gamma=0.05)
# lr_scheduler_D_B = torch.optim.lr_scheduler.StepLR(optimizer_D_B, step_size=20, gamma=0.05)

# Configure data loader
train_loader = DataLoader(
	customDataset(
		transform = transforms.Compose([
			transforms.Resize(opt.img_size), 
			transforms.ToTensor(), 
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		]),
		mode = 'train',
		txt_file = 'train_label.txt'
	),
	batch_size=opt.batch_size,
	shuffle=True,
	num_workers=opt.n_cpu,
	drop_last=True
)

img_shape = (opt.channels, opt.img_size, opt.img_size)
print("img_shape:"+str(img_shape))

FTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Target ground truths
valid = Variable(LTensor(opt.batch_size, 1).fill_(1), requires_grad=False).squeeze()
fake = Variable(LTensor(opt.batch_size, 1).fill_(0), requires_grad=False).squeeze()

# Loss visdom plot 
plotter = VisdomLinePlotter(env_name='main')

save_model_path = "save/trained_model"

# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
	for i, (imgA, imgB, lblA, lblB) in enumerate(train_loader):

		# Configure input
		real_A = Variable(imgA.type(FTensor))
		real_B = Variable(imgB.type(FTensor))

		age_A = Variable(lblA.type(FTensor))
		age_A = age_A.unsqueeze(1)
		age_A = age_A.repeat(1, opt.img_size**2)
		age_A = age_A.view(opt.batch_size, 1, opt.img_size, opt.img_size)

		age_B = Variable(lblB.type(FTensor))
		age_B = age_B.unsqueeze(1)
		age_B = age_B.repeat(1, opt.img_size**2)
		age_B = age_B.view(opt.batch_size, 1, opt.img_size, opt.img_size)

		# -----------------
		#  Train Generator
		# -----------------

		optimizer_G.zero_grad()

		# GAN loss
		gen_B = G_A2B(real_A, age_B)
		loss_GAN_A2B = criterion_GAN(D_B(gen_B, age_B), valid)

		gen_A = G_B2A(real_B, age_A)
		loss_GAN_B2A = criterion_GAN(D_A(gen_A, age_A), valid)

		total_GAN_loss = loss_GAN_A2B + loss_GAN_B2A

		# Cycle loss
		reconstruct_A = G_B2A(gen_B, age_A)
		loss_cycle_ABA = criterion_cycle(reconstruct_A, real_A)

		reconstruct_B = G_A2B(gen_A, age_B)
		loss_cycle_BAB = criterion_cycle(reconstruct_B, real_B)

		total_cycle_loss = loss_cycle_ABA + loss_cycle_BAB

		# Identity loss
		same_B = G_A2B(real_B, age_B)
		loss_identity_B = criterion_identity(same_B, real_B)

		same_A = G_B2A(real_A, age_A)
		loss_identity_A = criterion_identity(same_A, real_A)

		total_identity_loss = loss_identity_A + loss_identity_B

		# Total loss
		loss_G = total_GAN_loss + 10*total_cycle_loss + 5*total_identity_loss
		loss_G.backward()

		optimizer_G.step()

		# -----------------------
		#  Train Discriminator A
		# -----------------------

		optimizer_D_A.zero_grad()

		# Measure discriminator's ability to classify real from generated samples
		real_loss = criterion_GAN(D_A(real_A, age_A), valid)
		fake_loss = criterion_GAN(D_A(gen_A.detach(), age_A), fake)
		loss_D_A = (real_loss + fake_loss)
		loss_D_A.backward()

		optimizer_D_A.step()

		# -----------------------
		#  Train Discriminator B
		# -----------------------

		optimizer_D_B.zero_grad()

		# Measure discriminator's ability to classify real from generated samples
		real_loss = criterion_GAN(D_B(real_B, age_B), valid)
		fake_loss = criterion_GAN(D_B(gen_B.detach(), age_B), fake)
		loss_D_B = (real_loss + fake_loss) / 2 
		loss_D_B.backward()
		
		optimizer_D_B.step()

		# ----------------------
		#  Visualize and output
		# ----------------------
		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, %f]"
			% (epoch, opt.n_epochs, i, len(train_loader), loss_G.item(), loss_D_A.item(), loss_D_B.item()), "\t\t\r"
			, end=""
		)

		plotter.plot('loss', 'loss_G', 'Loss', epoch+i/len(train_loader), loss_G.item()/8)
		plotter.plot('loss', 'loss_D_A', 'Loss', epoch+i/len(train_loader), loss_D_A.item())
		plotter.plot('loss', 'loss_D_B', 'Loss', epoch+i/len(train_loader), loss_D_B.item())

		if i % 5*opt.sample_interval ==0:
			plotter.show(real_A.data[:1], real_B.data[:1], gen_A.data[:1], gen_B.data[:1], reconstruct_A[:1], reconstruct_B[:1])

	print()

	if epoch % opt.sample_interval == 0:   
		os.makedirs(save_model_path+"/%d" % epoch, exist_ok=True)
		torch.save(G_A2B.state_dict(), save_model_path+"/%d/G_A2B.pth" % epoch)
		torch.save(G_B2A.state_dict(), save_model_path+"/%d/G_B2A.pth" % epoch)
		torch.save(D_A.state_dict(), save_model_path+"/%d/D_A.pth" % epoch)
		torch.save(D_B.state_dict(), save_model_path+"/%d/D_B.pth" % epoch)

		# save_image(gen_A.data[:25], save_image_path+"/genA%d.png" % epoch, nrow=5, normalize=True)
		# save_image(gen_B.data[:25], save_image_path+"/genB%d.png" % epoch, nrow=5, normalize=True)
		# save_image(reconstruct_A.data[:25], save_image_path+"/recA%d.png" % epoch, nrow=5, normalize=True)
		# save_image(reconstruct_A.data[:25], save_image_path+"/recB%d.png" % epoch, nrow=5, normalize=True)
	
	# Update learning rates
	# lr_scheduler_G.step()
	# lr_scheduler_D_A.step()
	# lr_scheduler_D_B.step()
