"""
training.py

Methods for training (and validation) of EM electron model.
"""
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from torch.utils.data import Dataset
from torch.autograd import Variable
from PIL import Image

# Flag for data augmentation
augment = False
PIXEL_SIZE = 0.005
EVT_SIZE = 41                  # size of event grid (in X and Y)
ERR_SIZE = 84                 # size of error grid (in X and Y)
PIXEL_ERR_RANGE_MIN = -0.0025  # in-pixel error range minimum
PIXEL_ERR_RANGE_MAX = 0.0025   # in-pixel error range maximum

# Modified from medicaltorch.transforms: https://github.com/perone/medicaltorch/blob/master/medicaltorch/transforms.py
def rotate3D(data,axis=0):
    angle = np.random.uniform(-20,20)
    drot  = np.zeros(data.shape, dtype=data.dtype)

    for x in range(data.shape[axis]):
        if axis == 0:
            drot[x,:,:] = Image.fromarray(data[x,:,:]).rotate(angle,resample=False,expand=False,center=None,fillcolor=0)
        if axis == 1:
            drot[:,x,:] = Image.fromarray(data[:,x,:]).rotate(angle,resample=False,expand=False,center=None,fillcolor=0)
        if axis == 2:
            drot[:,:,x] = Image.fromarray(data[:,:,x]).rotate(angle,resample=False,expand=False,center=None,fillcolor=0)

    return drot

# Add a random gaussian noise to the data.
def gaussnoise(data, mean=0.0, stdev=0.05):

    dnoise = data + np.random.normal(loc=mean,scale=stdev,size=data.shape)
    return dnoise

# Sum the 3x3 square within the array containing the specified index and its neighbors.
# Set all neighbors included in the sum to 0 if the remove option is set.
def sum_neighbors(arr,ind,remove = False):

    # Start with the central pixel.
    sum = arr[ind]
    if(remove): arr[ind] = 0

    # Determine which neighbors exist.
    left_neighbor  = (ind[1]-1) >= 0
    right_neighbor = (ind[1]+1) < arr.shape[0]
    upper_neighbor = (ind[0]-1) >= 0
    lower_neighbor = (ind[0]+1) < arr.shape[1]

    # Add the 4 side-neighboring pixels to the sum.
    if(left_neighbor):
        sum += arr[ind[0],ind[1]-1]
        if(remove): arr[ind[0],ind[1]-1] = 0
    if(right_neighbor):
        sum += arr[ind[0],ind[1]+1]
        if(remove): arr[ind[0],ind[1]+1] = 0
    if(upper_neighbor):
        sum += arr[ind[0]-1,ind[1]]
        if(remove): arr[ind[0]-1,ind[1]] = 0
    if(lower_neighbor):
        sum += arr[ind[0]+1,ind[1]]
        if(remove): arr[ind[0]+1,ind[1]] = 0

    # Add the 4 diagonal neighbors to the sum.
    if(left_neighbor and upper_neighbor):
        sum += arr[ind[0]-1,ind[1]-1]
        if(remove): arr[ind[0]-1,ind[1]-1] = 0
    if(right_neighbor and upper_neighbor):
        sum += arr[ind[0]-1,ind[1]+1]
        if(remove): arr[ind[0]-1,ind[1]+1] = 0
    if(left_neighbor and lower_neighbor):
        sum += arr[ind[0]+1,ind[1]-1]
        if(remove): arr[ind[0]+1,ind[1]-1] = 0
    if(right_neighbor and lower_neighbor):
        sum += arr[ind[0]+1,ind[1]+1]
        if(remove): arr[ind[0]+1,ind[1]+1] = 0

    return sum


class EMDataset(Dataset):
    def __init__(self, dframe, noise_mean=0, noise_sigma=0, nstart=0, nend=0, add_noise=False, add_shift=0, augment=False):

        # Save some inputs for later use.
        self.dframe = dframe
        self.augment = augment
        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma
        self.add_noise = add_noise
        self.add_shift = add_shift

        # Open the dataframe.
        self.df_data = pd.read_pickle(dframe)

        # Extract the events array.
        self.events = self.df_data.event.unique()

        # Select the specified range [nstart:nend] for this dataset.
        if(nend == 0):
            self.events = self.events[nstart:]
            print("Created dataset for events from",nstart,"to",len(self.events))
        else:
            self.events = self.events[nstart:nend]
            print("Created dataset for events from",nstart,"to",nend)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):

        # Get the event ID corresponding to this key.
        evt = self.events[idx]

        # Prepare the event.
        evt_arr = np.zeros([101,101])
        df_evt = self.df_data[self.df_data.event == evt]
        for row,col,counts in zip(df_evt['row'].values,df_evt['col'].values,df_evt['counts'].values):
            evt_arr[row,col] += counts

        evt_arr = evt_arr[50-int((EVT_SIZE-1)/2):50+int((EVT_SIZE-1)/2)+1,50-int((EVT_SIZE-1)/2):50+int((EVT_SIZE-1)/2)+1]

        # Normalize to value of greatest magnitude = 1.
        #evt_arr /= 10000 #np.max(np.abs(evt_arr))

        x_shift = 0.
        y_shift = 0.

        # Add the shift, if specified.
        if(self.add_shift > 0):

            x_shift = np.random.randint(-self.add_shift,self.add_shift)
            y_shift = np.random.randint(-self.add_shift,self.add_shift)

            evt_arr = np.roll(evt_arr,x_shift,axis=1)
            evt_arr = np.roll(evt_arr,y_shift,axis=0)

        # Add Gaussian noise.
        if(self.add_noise):
            evt_arr = gaussnoise(evt_arr, mean=self.noise_mean, stdev=self.noise_sigma)

        # Add the relative incident positions to the shifts.
        err = [PIXEL_SIZE*x_shift + df_evt.xinc.values[0], PIXEL_SIZE*y_shift + df_evt.yinc.values[0]]

        # Construct the error matrix.
        SHIFTED_ERR_RANGE_MIN = PIXEL_ERR_RANGE_MIN - self.add_shift*PIXEL_SIZE
        SHIFTED_ERR_RANGE_MAX = PIXEL_ERR_RANGE_MAX + self.add_shift*PIXEL_SIZE

        xbin = int(ERR_SIZE*(err[0] - SHIFTED_ERR_RANGE_MIN)/(SHIFTED_ERR_RANGE_MAX - SHIFTED_ERR_RANGE_MIN))
        xbin = max(xbin,0)
        xbin = min(xbin,ERR_SIZE-1)

        ybin = int(ERR_SIZE*(err[1] - SHIFTED_ERR_RANGE_MIN)/(SHIFTED_ERR_RANGE_MAX - SHIFTED_ERR_RANGE_MIN))
        ybin = max(ybin,0)
        ybin = min(ybin,ERR_SIZE-1)

        err_ind = (ybin*ERR_SIZE) + xbin

        return evt_arr,err,err_ind

# Custom batch function
def my_collate(batch):

    data,target = [], []
    for item in batch:
        d,t = item[0], item[2]

        # Apply a random transformation.
        if(augment):
            d = rotate3D(d)
        #
        # # Pad all 0s to get a time dimension of tdim.
        # d = np.pad(d, [(0,ch_dim-d.shape[0]),(0,0),(0,0)])

        #print("final shapes are",d.shape,t.shape)
        data.append(d)
        target.append(t)

    #print("Max in batch is",np.max(data))
    data = torch.tensor(data).float().unsqueeze(1)
    target = torch.tensor(np.array(target)).long()

    return (data, target)


def train(model, epoch, train_loader, optimizer):

    losses_epoch = []; accuracies_epoch = []
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        #print("Target is",target)

        output_score = model(data)
        m = nn.CrossEntropyLoss()
        loss = m(output_score,target)

        loss.backward()
        optimizer.step()

        maxvals = output_score.argmax(dim=1)
        correctvals = (maxvals == target)
        accuracy = correctvals.sum().float() / float(target.size(0))

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t score_max: {:.6f}\t score_min: {:.6f}; Accuracy {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), output_score.data.max(), output_score.data.min(), accuracy.data.item()))

        losses_epoch.append(loss.data.item())
        accuracies_epoch.append(accuracy.data.item())

    print("---EPOCH AVG TRAIN LOSS:",np.mean(losses_epoch),"ACCURACY:",np.mean(accuracies_epoch))
    with open("train.txt", "a") as ftrain:
        ftrain.write("{} {} {}\n".format(epoch,np.mean(losses_epoch),np.mean(accuracies_epoch)))

def val(model, epoch, val_loader):

    losses_epoch = []; accuracies_epoch = []
    for batch_idx, (data, target) in enumerate(val_loader):

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output_score = model(data)
        m = nn.CrossEntropyLoss()
        loss = m(output_score,target)

        maxvals = output_score.argmax(dim=1)
        correctvals = (maxvals == target)
        accuracy = correctvals.sum().float() / float(target.size(0))

        if batch_idx % 1 == 0:
            print('--Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t score_max: {:.6f}\t score_min: {:.6f}; Accuracy {:.3f}'.format(
                epoch, batch_idx * len(data), len(val_loader.dataset),
                100. * batch_idx / len(val_loader), loss.data.item(), output_score.data.max(), output_score.data.min(), accuracy.data.item()))

        losses_epoch.append(loss.data.item())
        accuracies_epoch.append(accuracy.data.item())

    print("---EPOCH AVG VAL LOSS:",np.mean(losses_epoch),"ACCURACY:",np.mean(accuracies_epoch))
    with open("val.txt", "a") as fval:
        fval.write("{} {} {}\n".format(epoch,np.mean(losses_epoch),np.mean(accuracies_epoch)))


# OLD CODE
# if(self.add_noise):
#
#     max_before_noise = np.unravel_index(evt_arr.argmax(),evt_arr.shape)
#
#     # Add the noise.
#     evt_arr = gaussnoise(evt_arr, mean=self.noise_mean, stdev=self.noise_sigma)
#
#     # Make a copy of the array that we can safely modify.
#     evt_arr_temp = np.copy(evt_arr)
#
#     # ------------------------------------------------------------------------
#     # Determine the new maximum, as the largest 3x3 sum around a maximum pixel.
#     # 1. Find the initial maximum and compute sum of surrounding 3x3 region
#     # 2. Remove 3x3 region summed in step 1 from consideration for being maximum pixel
#     # 3. Find a new maximum and compute the 3x3 sum about this new maximum
#     # 4. Remove 3x3 region summed in step 2 from consideration for being maximum pixel
#     # 5. If the 3x3 region sum from step 3 is greater than that of the initial maximum from step 1, replace the initial maximum with the new maximum
#     # 6. Repeat steps 3-5 until the new maximum is <= 0 or the region sum is less than the initial region sum
#
#     # Get the initial maximum and neighbor sum, removing these neighbors from consideration for the next maximum.
#     max_init   = np.unravel_index(evt_arr_temp.argmax(),evt_arr.shape)
#     nbsum_init = sum_neighbors(evt_arr_temp,max_init,remove=True)
#     found = False
#     while(not found):
#
#         # Get the next maximum.
#         max_current   = np.unravel_index(evt_arr_temp.argmax(),evt_arr.shape)
#         nbsum_current = sum_neighbors(evt_arr,max_current,remove=False)        # note: the sum should be from the original (unmodified) array
#
#         # A maximum of less than or equal to zero means we are done, and we should keep the previous maximum.
#         if(evt_arr[max_current] <= 0):
#             found = True
#
#         # If the current neighbor sum is greater than that of the initial maximum, replace the initial maximum with the current one.
#         elif(nbsum_current > nbsum_init):
#             sum_neighbors(evt_arr_temp,max_current,remove=True)  # remove the neighbors of the current maximum
#
#             # Replace the initial maximum and its neighbor sum.
#             #print("Replacing init maximum at",max_init,"with max_current",max_current)
#             max_init = max_current
#             nbsum_init = nbsum_current
#
#         # Otherwise keep the current initial maximum.
#         else:
#             found = True
#     # ------------------------------------------------------------------------
#
#     # Calculate the shift.
#     x_shift = (max_init[1] - max_before_noise[1])
#     y_shift = (max_init[0] - max_before_noise[0])
#
#     # Shift to the new maximum.
#     evt_arr = np.roll(evt_arr,x_shift,axis=1)
#     evt_arr = np.roll(evt_arr,y_shift,axis=0)
