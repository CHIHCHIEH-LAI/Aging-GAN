import numpy as np
import math
import os
import pdb
import torch
from torchvision import transforms
from visdom import Visdom

class VisdomLinePlotter(object):
	"""Plots to Visdom"""
	def __init__(self, port_num=8800, env_name='main'):
		self.viz = Visdom(port=port_num)
		self.env = env_name
		self.plots = {}
		self.de_norm = transforms.Normalize(
			   mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
				std=[1/0.5, 1/0.5, 1/0.5]
			)
	def plot(self, var_name, split_name, title_name, x, y):
		if not hasattr(self, 'plot_data'):
			self.plot_data = {'X': [], 'Y': [], 'legend': []}


		if var_name not in self.plots:
			self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
				legend=[split_name],
				title=title_name,
				xlabel='Epochs',
				ylabel=var_name
			))
		else:
			self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

	def show(self, realA_list, realB_list, genA_list, genB_list, reconA_list, reconB_list):
		
		realA_list = self.denormalize(realA_list)
		realB_list = self.denormalize(realB_list)
		genA_list = self.denormalize(genA_list)
		genB_list = self.denormalize(genB_list)
		reconA_list = self.denormalize(reconA_list)
		reconB_list = self.denormalize(reconB_list)
		img_list = torch.cat((realA_list, genB_list, reconA_list, realB_list, genA_list, reconB_list))

		# pdb.set_trace()
		# captions = "ageA=%d, ageB=%d" %(ageA, ageB)
		self.viz.images(img_list, opts=dict(title='imgs'), nrow = 3, win = 1)
		# self.viz.images(realB_list, opts=dict(title='real_B'), win = 2)
		# self.viz.images(genA_list, opts=dict(title='gen_A'), win = 3)
		# self.viz.images(genB_list, opts=dict(title='gen_B'), win = 4)


	def denormalize(self, img_list):
		output = torch.zeros(img_list.shape)
		for idx in range(img_list.shape[0]):
			output[idx] = self.de_norm(img_list[idx])

		return output*255