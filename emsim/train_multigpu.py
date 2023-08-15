"""
train_multigpu.py

Script for training on multiple GPUs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import training as tr
import emsim.emnet as emnet
import emsim_utils

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

from unet import UNet
import scipy.optimize as optimize
from scipy.optimize import curve_fit

#
modeldir = '/global/cscratch1/sd/jrenner1/models'
lrate       = 1e-3   # Learning rate to use in the training.
load_model  = False   # Load an existing model
tr.augment  = False  # Enable/disable data augmentation
epoch_start = 0      # Number of initial epoch
epoch_end   = 600    # Number of final epoch
model_load_checkpoint = "{}/model_10cells_noise_100k_74.pt".format(modeldir)

# Create the datasets.
dataset_all   = tr.EMDataset("/global/cscratch1/sd/jrenner1/data/EM_4um_back_10M_300keV.pkl",noise_mean=0,noise_sigma=20,add_noise=True,add_shift=0)
dataset_train = tr.EMDataset("/global/cscratch1/sd/jrenner1/data/EM_4um_back_10M_300keV.pkl",noise_mean=0,noise_sigma=20,add_noise=True,nstart=20000,nend=100000,add_shift=0)
dataset_val   = tr.EMDataset("/global/cscratch1/sd/jrenner1/data/EM_4um_back_10M_300keV.pkl",noise_mean=0,noise_sigma=20,add_noise=True,nstart=0,nend=20000,add_shift=0)

# Create the loaders.
train_loader = DataLoader(dataset_train, batch_size=1000, shuffle=True, collate_fn=tr.my_collate, num_workers=32)
val_loader = DataLoader(dataset_val, batch_size=1000, shuffle=True, collate_fn=tr.my_collate, num_workers=32)
#test_loader = DataLoader(dataset_test, batch_size=15, shuffle=True, collate_fn=tr.my_collate, num_workers=4)

# Define the model.
#model = emnet.FCNet()
model = emnet.basicCNN()
#model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lrate, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01, amsgrad=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Load the model from file.
if(load_model):
    model.load_state_dict(torch.load(model_load_checkpoint))
    #model.load_state_dict(torch.load(model_load_checkpoint,map_location=torch.device('cpu')))
    model.eval()

# Run the training.
for epoch in range(epoch_start,epoch_end):
    print("Epoch: ",epoch)
    model.train()
    train_loss = tr.train(model, epoch, train_loader, optimizer)
    scheduler.step(train_loss)
    with torch.no_grad():
        model.eval()
        val_loss = tr.val(model, epoch, val_loader)
        #scheduler.step(val_loss)
    torch.save(model.state_dict(), "{}/model_init_{}.pt".format(modeldir,epoch))
