import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import shutil

from torch.autograd import Variable
from torch.utils import data
import os

from dataset import IC19Loader
from metrics import runningScore
import models
from util import Logger, AverageMeter
import time
import util

data_loader = IC19Loader(is_transform=True, img_size=640, kernel_num=7, min_scale=720)
train_loader = torch.utils.data.DataLoader(
	data_loader,
	batch_size=1,
	shuffle=True,
	drop_last=True,
	pin_memory=True)

for i in range(5):
	for batch_idx, (imgs, gt_texts, gt_kernels, training_masks) in enumerate(train_loader):
		print(imgs.shape,gt_kernels.shape,gt_texts.shape)
