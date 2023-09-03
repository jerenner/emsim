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

import emsim.emnet as emnet

# Flag for data augmentation
augment = False

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

def create_L_event():
    evt_arr = np.zeros([101,101])

    horiz = np.random.rand() > 0.5  # long leg of L is horizontal (over columns)
    dir1  = np.random.rand() > 0.5  # direction of long leg of L
    dir2  = np.random.rand() > 0.5  # direction of short leg of L

    # Choose the random values, and normalize to 1.
    vals = 0.5*np.random.rand(4) + 0.5
    vals /= np.sum(vals)

    # Flip the direction of the long leg if dir1 = True
    sign1 = 1
    if(dir1): sign1 = -1

    # Draw the L
    if(horiz):
        for i in range(len(vals)-1):
            evt_arr[50,50+sign1*i] = vals[i]
        if(dir2):
            evt_arr[49,50+sign1*(len(vals)-2)] = vals[len(vals)-1]
        else:
            evt_arr[51,50+sign1*(len(vals)-2)] = vals[len(vals)-1]
    else:
        for i in range(len(vals)-1):
            evt_arr[50+sign1*i,50] = vals[i]
        if(dir2):
            evt_arr[50+sign1*(len(vals)-2),49] = vals[len(vals)-1]
        else:
            evt_arr[50+sign1*(len(vals)-2),51] = vals[len(vals)-1]

    # Return the event.
    return evt_arr

class EMDataset(Dataset):
    def __init__(self, dframe, noise_mean=0, noise_sigma=0, nstart=0, nend=0, add_noise=False, add_shift=-1, augment=False, Ltest = False):

        # Save some inputs for later use.
        self.dframe = dframe
        self.augment = augment
        self.Ltest = Ltest
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
        df_evt = self.df_data[self.df_data.event == evt]

        # Prepare the event.
        if(self.Ltest):
            evt_arr = create_L_event()
        else:
            evt_arr = np.zeros([101,101])
            for row,col,counts in zip(df_evt['row'].values,df_evt['col'].values,df_evt['counts'].values):
                evt_arr[row,col] += counts

        # Use a windowed event+noise to determine the maximum pixel if no manual shift specified.
        if(self.add_shift < 0):

            ssize = 101
            evt_small = evt_arr[50-int((ssize-1)/2):50+int((ssize-1)/2)+1,50-int((ssize-1)/2):50+int((ssize-1)/2)+1]
            if(self.add_noise):
                evt_small = gaussnoise(evt_small, mean=self.noise_mean, stdev=self.noise_sigma)
            yx_shift = np.unravel_index(np.argmax(evt_small),evt_small.shape)
            y_shift = yx_shift[0] - int((ssize-1)/2)
            x_shift = yx_shift[1] - int((ssize-1)/2)
            #print("Argmax was {}".format(yx_shift))
            #print("Found x-shift = {}, y-shift = {}".format(x_shift,y_shift))
        else:
            x_shift = 0
            y_shift = 0

        # Extract the specified event size from the larger event, centered on the maximum pixel.
        evt_arr = evt_arr[50+y_shift-int((emnet.EVT_SIZE-1)/2):50+y_shift+int((emnet.EVT_SIZE-1)/2)+1,50+x_shift-int((emnet.EVT_SIZE-1)/2):50+x_shift+int((emnet.EVT_SIZE-1)/2)+1]

        # Normalize to value of greatest magnitude = 1.
        #evt_arr /= 10000 #np.max(np.abs(evt_arr))

        # Add a manual shift, if specified.
        if(self.add_shift > 0):

            x_shift = np.random.randint(-self.add_shift,self.add_shift)
            y_shift = np.random.randint(-self.add_shift,self.add_shift)

            evt_arr = np.roll(evt_arr,x_shift,axis=1)
            evt_arr = np.roll(evt_arr,y_shift,axis=0)

        # Add Gaussian noise.
        if(self.add_noise):
            evt_arr = gaussnoise(evt_arr, mean=self.noise_mean, stdev=self.noise_sigma)

        # Add the relative incident positions to the shifts.
        if(self.Ltest):
            err = [-emnet.PIXEL_SIZE*x_shift, -emnet.PIXEL_SIZE*y_shift]
        else:
            err = [-emnet.PIXEL_SIZE*x_shift + df_evt.xinc.values[0], -emnet.PIXEL_SIZE*y_shift + df_evt.yinc.values[0]]
        #err = [df_evt.xinc.values[0], df_evt.yinc.values[0]]

        # Construct the error matrix.
        SHIFTED_ERR_RANGE_MIN = emnet.PIXEL_ERR_RANGE_MIN - self.add_shift*emnet.PIXEL_SIZE
        SHIFTED_ERR_RANGE_MAX = emnet.PIXEL_ERR_RANGE_MAX + self.add_shift*emnet.PIXEL_SIZE

        xbin = int(emnet.ERR_SIZE*(err[0] - SHIFTED_ERR_RANGE_MIN)/(SHIFTED_ERR_RANGE_MAX - SHIFTED_ERR_RANGE_MIN))
        xbin = max(xbin,0)
        xbin = min(xbin,emnet.ERR_SIZE-1)

        ybin = int(emnet.ERR_SIZE*(err[1] - SHIFTED_ERR_RANGE_MIN)/(SHIFTED_ERR_RANGE_MAX - SHIFTED_ERR_RANGE_MIN))
        ybin = max(ybin,0)
        ybin = min(ybin,emnet.ERR_SIZE-1)

        err_ind = (ybin*emnet.ERR_SIZE) + xbin

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


def my_collate_reg_line(batch):

    data,arg_max,evt_err,light_region = [], [], [], []
    for item in batch:
        d,a,ee,l = item[0], item[2], item[3], item[4]

        # Apply a random transformation.
        if(augment):
            d = rotate3D(d)
        #
        # # Pad all 0s to get a time dimension of tdim.
        # d = np.pad(d, [(0,ch_dim-d.shape[0]),(0,0),(0,0)])

        #print("final shapes are",d.shape,t.shape)
        data.append(d)
        arg_max.append(a)
        evt_err.append(ee)
        light_region.append(l)

    #print("Max in batch is",np.max(data))
    data = torch.tensor(data).float().unsqueeze(1)
    arg_max = torch.tensor(arg_max).int() #.unsqueeze(1)
    evt_err = torch.tensor(evt_err).float()
    light_region = torch.tensor(light_region).int()

    return (data, arg_max, evt_err, light_region)

def my_collate_reg_line_realdata(batch):

    data,arg_max,evt_err,light_region,line_m,line_b = [], [], [], [], [], []
    for item in batch:
        d,a,ee,l,m,b = item[0], item[2], item[3], item[4], item[5], item[6]

        # Apply a random transformation.
        if(augment):
            d = rotate3D(d)
        #
        # # Pad all 0s to get a time dimension of tdim.
        # d = np.pad(d, [(0,ch_dim-d.shape[0]),(0,0),(0,0)])

        #print("final shapes are",d.shape,t.shape)
        data.append(d)
        arg_max.append(a)
        evt_err.append(ee)
        light_region.append(l)
        line_m.append(m)
        line_b.append(b)

    #print("Max in batch is",np.max(data))
    data = torch.tensor(np.array(data)).float().unsqueeze(1)
    arg_max = torch.tensor(np.array(arg_max)).int() #.unsqueeze(1)
    evt_err = torch.tensor(np.array(evt_err)).float()
    light_region = torch.tensor(np.array(light_region)).int()
    line_m = torch.tensor(np.array(line_m)).float()
    line_b = torch.tensor(np.array(line_b)).float()

    return (data, arg_max, evt_err, light_region, line_m, line_b)

def my_collate_unet(batch):

    data,target = [], []
    for item in batch:
        d,t = item[0], item[1]

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
    target = torch.tensor(target).float() #.unsqueeze(1)

    return (data, target)

class RealFrameDataset(Dataset):
    def __init__(self, levents_file, revents_file, istart = 0, nframes = 500000):

        # Save the file names and number of frames.
        self.levents_file = levents_file
        self.revents_file = revents_file
        self.nframes = nframes
        self.istart = istart

        # Load the event arrays for light region 0 (below the line, or on the "right").
        f_revents = np.load(revents_file)
        self.revents_frame = f_revents['valid_subimages'][istart:istart+nframes]
        self.revents_frame_c = f_revents['valid_subimages_c'][istart:istart+nframes]
        self.revents_m     = f_revents['line_m'][istart:istart+nframes]
        self.revents_b     = f_revents['line_b'][istart:istart+nframes]
        self.revents_i     = np.arange(nframes)
        np.random.shuffle(self.revents_i)

        # Load the event arrays for light region 1 (above the line, or on the "left").
        f_levents = np.load(levents_file)
        self.levents_frame = f_levents['valid_subimages'][istart:istart+nframes]
        self.levents_frame_c = f_levents['valid_subimages_c'][istart:istart+nframes]
        self.levents_m     = f_levents['line_m'][istart:istart+nframes]
        self.levents_b     = f_levents['line_b'][istart:istart+nframes]
        self.levents_i     = np.arange(nframes)
        np.random.shuffle(self.levents_i)

    def __len__(self):
        return self.nframes

    def __getitem__(self, idx):

        return self.get_reg_line_event(idx)

    def get_reg_line_event(self, idx):

        # Determine the location of the light region, below (= 0), or above (= 1) the line.
        light_region = np.random.randint(2)

        if(light_region == 0):
            # iframe = self.revents_i[idx]
            # frame = self.revents_frame[iframe]
            # frame_cmax = self.revents_frame_c[iframe]
            # line_m = self.revents_m[iframe]
            # line_b = self.revents_b[iframe]

            # Attempt to symmetrize left events
            iframe = self.levents_i[idx]
            frame = np.copy(np.flip(self.levents_frame[iframe]))
            frame_cmax = np.copy(np.flip(self.levents_frame_c[iframe]))
            line_m = 1*self.levents_m[iframe]
            line_b = 2*5.5 - 2*self.levents_m[iframe]*5.5 - self.levents_b[iframe]
        else:
            iframe = self.levents_i[idx]
            frame = self.levents_frame[iframe]
            frame_cmax = self.levents_frame_c[iframe]
            line_m = self.levents_m[iframe]
            line_b = self.levents_b[iframe]

        # # Shift the frame so that it's centered on the max pixel.
        # frame_cmax = np.zeros(frame.shape)
        #

        # Get the maximum pixel.
        arg_max = np.unravel_index(np.argmax(frame),frame.shape)

        #
        # # Construct the shifted frame.
        # delta = int((frame.shape[0]-1)/2)  # the extent of the event from the center pixel
        # ileft = max(arg_max[0]-delta,0); delta_ileft = arg_max[0] - ileft
        # jleft = max(arg_max[1]-delta,0); delta_jleft = arg_max[1] - jleft
        #
        # iright = min(arg_max[0]+delta+1,frame.shape[0]); delta_iright = iright - arg_max[0]
        # jright = min(arg_max[1]+delta+1,frame.shape[1]); delta_jright = jright - arg_max[1]
        # frame_cmax[delta-delta_ileft:delta+delta_iright,delta-delta_jleft:delta+delta_jright] += frame[ileft:iright,jleft:jright]

        # Return the frame centered on the max value, the shifted frame, tha maximum shift, and the event error ([x,y], in mm).
        evt_err = np.zeros(2)
        return frame_cmax, frame, arg_max, evt_err, light_region, line_m, line_b


class EMFrameDataset(Dataset):
    def __init__(self, emdset, nframes=1000, frame_size=576, nelec_mean=2927.294, nelec_sigma=70.531, noise_mean=0, noise_sigma=0, m_line=None, b_line=None, th_classical = 825, lside = -1, res_factor = 1):

        # Save some inputs for later use.
        self.emdset = emdset
        self.nframes = nframes
        self.frame_size = frame_size
        self.nelec_mean = nelec_mean
        self.nelec_sigma = nelec_sigma
        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma
        self.m_line = m_line
        self.b_line = b_line
        self.th_classical = th_classical
        self.lside = lside
        self.res_factor = res_factor     # resolution factor: prediction grid has resolution of event grid * this factor; for now, must be odd

        # Get the row and column indices.
        indices = np.indices((frame_size,frame_size))
        self.irows = indices[0]
        self.icols = indices[1]

    def __len__(self):
        return self.nframes

    def __getitem__(self, idx):

        return self.get_reg_line_event(idx)
        #return self.get_reg_event(idx)
        #return self.get_hg_event(idx)

    def get_reg_line_event(self, idx):

        # Create a random event (index does nothing, though could possibly be used as seed).
        frame = np.zeros([self.frame_size,self.frame_size])

        # Determine the location of the light region, below (= 0), or above (= 1) the line.
        if(self.lside >= 0):
            light_region = self.lside
        else:
            light_region = np.random.randint(2)

        # Add all electrons to the event.
        nelec = 1
        iel = 0
        evt_err = None
        while(iel < nelec):
        #for iel in range(nelec):

            # Pick a random location in the frame for the electron.
            loc_chosen = False
            while(not loc_chosen):
                eloc = np.unravel_index(np.random.randint(frame.size),frame.shape)
                if(eloc[0] > 1 and eloc[0] < frame.shape[0]-2 and eloc[1] > 1 and eloc[1] < frame.shape[1]-2):
                    loc_chosen = True

            # Use the central location in the frame for the electron.
            #eloc = np.unravel_index(int((frame.size-1)/2),frame.shape)
            #print("Loc is ",eloc,"for frame size",frame.size,"and frame shape",frame.shape)

            #ievt = idx
            # Pick a random event from the EM dataset.
            ievt = np.random.randint(len(self.emdset))

            evt_item = self.emdset[ievt]
            evt_arr = evt_item[0]
            evt_err = evt_item[1]

            # ------------------------------------------------------------------
            # Throw an electron.

            # Ensure the error is within range.
            err_max = int(frame.shape[0]-1)/2*emnet.PIXEL_SIZE
            #err_max = 1.5*emnet.PIXEL_SIZE
            if(abs(evt_err[0]) > err_max or abs(evt_err[1]) > err_max):
                #print("Throwing event with error",evt_err)
                continue

            # If we have specified an edge, check whether we should throw the electron.
            if((self.m_line is not None and self.b_line is not None)):

                # Do not throw the electron in the dark region.
                row_true = eloc[0] + evt_err[1]/emnet.PIXEL_SIZE + 0.5
                col_true = eloc[1] + evt_err[0]/emnet.PIXEL_SIZE + 0.5
                if(((light_region == 0) and (row_true < self.m_line*col_true + self.b_line)) or ((light_region == 1) and (row_true > self.m_line*col_true + self.b_line))):
                    continue

            # Consider the electron added.
            iel += 1

            # Add the electron to the frame.
            delta = int((emnet.EVT_SIZE-1)/2)  # the extent of the event from the center pixel
            ileft = max(eloc[0]-delta,0); delta_ileft = eloc[0] - ileft
            jleft = max(eloc[1]-delta,0); delta_jleft = eloc[1] - jleft

            iright = min(eloc[0]+delta+1,frame.shape[0]); delta_iright = iright - eloc[0]
            jright = min(eloc[1]+delta+1,frame.shape[1]); delta_jright = jright - eloc[1]

            frame[ileft:iright,jleft:jright] += evt_arr[delta-delta_ileft:delta+delta_iright,delta-delta_jleft:delta+delta_jright] #/10000

        # Shift the frame so that it's centered on the max pixel.
        frame_cmax = np.zeros([self.frame_size,self.frame_size])

        # Get the maximum pixel.
        arg_max = np.unravel_index(np.argmax(frame),frame.shape)

        # Construct the shifted frame.
        delta = int((emnet.EVT_SIZE-1)/2)  # the extent of the event from the center pixel
        ileft = max(arg_max[0]-delta,0); delta_ileft = arg_max[0] - ileft
        jleft = max(arg_max[1]-delta,0); delta_jleft = arg_max[1] - jleft

        iright = min(arg_max[0]+delta+1,frame.shape[0]); delta_iright = iright - arg_max[0]
        jright = min(arg_max[1]+delta+1,frame.shape[1]); delta_jright = jright - arg_max[1]
        frame_cmax[delta-delta_ileft:delta+delta_iright,delta-delta_jleft:delta+delta_jright] += frame[ileft:iright,jleft:jright]

        # Return the frame centered on the max value, the shifted frame, tha maximum shift, and the event error ([x,y], in mm).
        return frame_cmax, frame, arg_max, evt_err, light_region

    def get_reg_event(self, idx):

        # Create a random event (index does nothing, though could possibly be used as seed).
        frame = np.zeros([self.frame_size,self.frame_size])

        # Add all electrons to the event.
        nelec = 1
        iel = 0
        while(iel < nelec):
        #for iel in range(nelec):

            # Pick a random location in the frame for the electron.
            #eloc = np.unravel_index(np.random.randint(frame.size),frame.shape)

            # Use the central location in the frame for the electron.
            eloc = np.unravel_index(int((frame.size-1)/2),frame.shape)
            #print("Loc is ",eloc,"for frame size",frame.size,"and frame shape",frame.shape)

            #ievt = idx
            # Pick a random event from the EM dataset.
            ievt = np.random.randint(len(self.emdset))

            evt_item = self.emdset[ievt]
            evt_arr = evt_item[0]
            evt_err = evt_item[1]

            # Throw an electron.
            #err_max = int(frame.shape[0]-1)/2*emnet.PIXEL_SIZE
            err_max = 1.5*emnet.PIXEL_SIZE
            if(abs(evt_err[0]) > err_max or abs(evt_err[1]) > err_max):
                #print("Throwing event with error",evt_err)
                continue
            iel += 1

            # Add the electron to the frame.
            delta = int((emnet.EVT_SIZE-1)/2)  # the extent of the event from the central pixel
            ileft = max(eloc[0]-delta,0); delta_ileft = eloc[0] - ileft
            jleft = max(eloc[1]-delta,0); delta_jleft = eloc[1] - jleft

            iright = min(eloc[0]+delta+1,frame.shape[0]); delta_iright = iright - eloc[0]
            jright = min(eloc[1]+delta+1,frame.shape[1]); delta_jright = jright - eloc[1]

            frame[ileft:iright,jleft:jright] += evt_arr[delta-delta_ileft:delta+delta_iright,delta-delta_jleft:delta+delta_jright] #/10000

            # Get the maximum pixel.
            arg_max = np.unravel_index(np.argmax(frame),frame.shape)

            # Compute the vector from the max point to the true point.
            #vec_row = ((arg_max[0] - eloc[0])*emnet.PIXEL_SIZE + evt_err[1])/(emnet.PIXEL_SIZE) # y-error
            #vec_col = ((arg_max[1] - eloc[1])*emnet.PIXEL_SIZE + evt_err[0])/(emnet.PIXEL_SIZE) # x-error
            vec_row = evt_err[1]/emnet.PIXEL_SIZE
            vec_col = evt_err[0]/emnet.PIXEL_SIZE
            if(not(arg_max[0] == 5 and arg_max[1] == 5)):
                print("Loc is ",eloc,"for frame size",frame.size,"and frame shape",frame.shape)
                print("Initial event max is",np.unravel_index(np.argmax(evt_arr),evt_arr.shape))
                print("Arg max is",arg_max)
            # print("Eloc = ",eloc)
            # print("Evt error is",evt_err)
            # print("Truth is: row =",vec_row,", col =",vec_col)

            # Create the vector truth.
            #vec_truth = [(eloc[0] + 0.5)*emnet.PIXEL_SIZE + evt_err[0], (eloc[1] + 0.5)*emnet.PIXEL_SIZE + evt_err[1]] # absolute coord.
            vec_truth = [vec_row, vec_col]
            #err_rng = 2*err_max
            #vec_truth = [(vec_row + err_max)/err_rng, (vec_col + err_max)/err_rng]
            #print("Final error is",vec_truth)

            #print("Threw electron with error from central pixel as: ",vec_truth)

        return frame, vec_truth

    def get_hg_event(self, idx):

        # Create a random event (index does nothing, though could possibly be used as seed).
        frame = np.zeros([self.frame_size,self.frame_size])
        frame_truth = np.zeros(frame.shape)

        # ----------------------------------------------------------------------
        # Create the array for the high-resolution-grid truth.
        rfac = self.res_factor
        frame_hrg_truth = np.zeros([self.frame_size*rfac,self.frame_size*rfac])

        hrg_frame = np.zeros([self.frame_size*rfac, self.frame_size*rfac])
        hrg_indices = np.indices((self.frame_size*rfac,self.frame_size*rfac))
        irows_hrg = hrg_indices[0]
        icols_hrg = hrg_indices[1]

        # Determine the number of electrons.
        nelec = 1 #int(np.random.normal(loc=self.nelec_mean,scale=self.nelec_sigma))

        # Determine the location of the light region, below (= 0), or above (= 1) the line.
        if(self.lside >= 0):
            light_region = self.lside
        else:
            light_region = np.random.randint(2)

        # Create the edge truth.
        if(light_region == 0):
            edge_truth = irows_hrg >= self.m_line*icols_hrg + self.b_line*rfac
        else:
            edge_truth = irows_hrg <= self.m_line*icols_hrg + self.b_line*rfac

        # Add all electrons to the event.
        iel = 0
        while(iel < nelec):
        #for iel in range(nelec):

            # Pick a random location in the high-resolution frame for the electron.
            eloc_hrg = np.unravel_index(np.random.randint(hrg_frame.size),hrg_frame.shape)

            # If we have specified an edge, check whether we should throw the electron.
            if((self.m_line is not None and self.b_line is not None)):

                # Do not throw the electron in the dark region.
                irow = irows_hrg[eloc_hrg]
                icol = icols_hrg[eloc_hrg]
                if(((light_region == 0) and (irow < self.m_line*icol + self.b_line*rfac)) or ((light_region == 1) and (irow > self.m_line*icol + self.b_line*rfac))):
                    continue

            # Throw an electron.
            iel += 1

            # Convert the location on the HRG to the original grid.
            eloc = (int(eloc_hrg[0]/rfac),int(eloc_hrg[1]/rfac))

            # Pick a random event from the EM dataset.
            ievt = np.random.randint(len(self.emdset))
            evt_item = self.emdset[ievt]
            evt_arr = evt_item[0]
            evt_err = evt_item[1]

            # Add the electron to the frame.
            delta = int((emnet.EVT_SIZE-1)/2)  # the extent of the event from the central pixel
            ileft = max(eloc[0]-delta,0); delta_ileft = eloc[0] - ileft
            jleft = max(eloc[1]-delta,0); delta_jleft = eloc[1] - jleft

            iright = min(eloc[0]+delta+1,frame.shape[0]); delta_iright = iright - eloc[0]
            jright = min(eloc[1]+delta+1,frame.shape[1]); delta_jright = jright - eloc[1]

            frame[ileft:iright,jleft:jright] += evt_arr[delta-delta_ileft:delta+delta_iright,delta-delta_jleft:delta+delta_jright]

            # Add the electron to the truth array.
            frame_truth[eloc] = 1

            # Add the electron to the high-res truth array.
            frame_hrg_truth[eloc_hrg] = 1

            # rowoffset = int(rfac*(evt_err[0] + emnet.PIXEL_SIZE/2)/emnet.PIXEL_SIZE)
            # coloffset = int(rfac*(evt_err[1] + emnet.PIXEL_SIZE/2)/emnet.PIXEL_SIZE)
            # hrg_eloc = (rfac*eloc[0] + rowoffset, rfac*eloc[1] + coloffset)
            # print("For eloc",eloc,"got final loc",hrg_eloc,"plus offsets row=",rowoffset,"and col=",coloffset,"with rowerr=",evt_err[0],"and colerr=",evt_err[1])


        # Add the noise.
        if(self.noise_sigma > 0):
            frame = gaussnoise(frame, mean=self.noise_mean, stdev=self.noise_sigma)

        # Compute the distance matrix.
        #dist = (self.m_line*self.icols - self.irows + self.b_line) / (self.m_line**2 + 1)

        # Resample the frame to higher resolution.
        # for row in range(frame.shape[0]):
        #     for col in range(frame.shape[1]):
        #         uniform_dist = np.ones([rfac,rfac]) #np.random.random_sample([rfac,rfac])
        #         uniform_dist = frame[row,col]*uniform_dist/np.sum(uniform_dist)
        #         hrg_frame[row*rfac:(row+1)*rfac,col*rfac:(col+1)*rfac] = uniform_dist

        # ----------------------------------------------------------------------
        # Resample the frame to higher resolution.
        for row in range(rfac):
            for col in range(rfac):
                hrg_frame[row::rfac,col::rfac] = frame/(rfac*rfac)

        # Create the threshold-based "truth".
        #th_truth = (frame > self.th_classical)
        #edge_frame = frame * edge_truth

        # Include the edge information.
        # Remove noise to reduce bias in 3x3 CM determination.
        edge_frame = (hrg_frame - self.noise_mean/(rfac*rfac)) * edge_truth

        # ----------------------------------
        # Perform a 3*rfac x 3*rfac average.
        # ----------------------------------
        # Get the maximum argument.
        args_max = np.argwhere(edge_frame == np.amax(edge_frame))
        arg_max = args_max[int(len(args_max)/2)]
        #arg_max = np.unravel_index(np.argmax(edge_frame),edge_frame.shape)
        # print("Arg max is",arg_max)

        # Get the bounding box.
        llimit = int((3*rfac-1)/2)
        rlimit = int((3*rfac-1)/2 + 1)
        r_lbound = max(arg_max[0]-llimit,0)
        r_rbound = min(arg_max[0]+rlimit,edge_frame.shape[0])
        c_lbound = max(arg_max[1]-llimit,0)
        c_rbound = min(arg_max[1]+rlimit,edge_frame.shape[1])
        max_3x3 = edge_frame[r_lbound:r_rbound,c_lbound:c_rbound]
        # print("max_3x3 is",max_3x3)
        # print("llimit =",llimit," and rlimit =",rlimit)
        # print("r_lbound = ",r_lbound," and r_rbound = ",r_rbound)
        # print("c_lbound = ",c_lbound," and c_rbound = ",c_rbound)

        # Zero-out all negative numbers.
        max_3x3[max_3x3 < 0] = 0

        # Compute the offsets - note the meshgrid takes (x,y) = (col,row) instead of (row,col).
        coords_pixels_3x3 = np.meshgrid(np.arange(c_rbound-c_lbound),np.arange(r_rbound-r_lbound))
        # print("coords_pixels_3x3[0]",coords_pixels_3x3[0])
        # print("coords_pixels_3x3[1]",coords_pixels_3x3[1])
        row_offset = np.rint(np.sum(coords_pixels_3x3[1]*max_3x3)/np.sum(max_3x3)).astype('int') - (arg_max[0] - r_lbound)
        col_offset = np.rint(np.sum(coords_pixels_3x3[0]*max_3x3)/np.sum(max_3x3)).astype('int') - (arg_max[1] - c_lbound)
        # print("row_offset=",row_offset,"and col_offset=",col_offset)
        # print("arg_max[0]=",arg_max[0],"and arg_max[1]=",arg_max[1])
        # print("max_3x3 sum=",np.sum(max_3x3))
        # print("weighted sum row=",np.sum(coords_pixels_3x3[1]*max_3x3),"norm=",np.sum(coords_pixels_3x3[1]*max_3x3)/np.sum(max_3x3))
        # print("weighted sum col=",np.sum(coords_pixels_3x3[0]*max_3x3),"norm=",np.sum(coords_pixels_3x3[0]*max_3x3)/np.sum(max_3x3))

        # Set the pixel in the truth.
        th_truth = np.zeros(edge_frame.shape)
        th_truth[arg_max[0] + row_offset, arg_max[1] + col_offset] = 1
        #th_truth[arg_max[0], arg_max[1]] = 1  # for normal maximum-finding

        # (Use the maximum)
        # th_truth[arg_max] = 1

        # Compute the distance matrix.
        dist = (self.m_line*icols_hrg - irows_hrg + self.b_line*rfac) / (self.m_line**2 + 1)

        # Store all the truth matrices in a single matrix.
        all_truth = []
        all_truth.append(frame_hrg_truth) #(frame_truth)
        all_truth.append(th_truth)
        all_truth.append(edge_truth)
        all_truth.append(dist)
        all_truth = np.array(all_truth)

        return hrg_frame,all_truth
        #return frame,all_truth

def loss_reg_edge(evt_arr, evt_err, output, row_coords, col_coords, arg_max, line_m, line_b, light_region, epoch = 0, sigma_dist = 1.0, w_edge = 100):

    # Compute the "error" (vector from center of max pixel) in row and column.
    col_err = output[:,0]
    row_err = output[:,1]
    # print("Output is (col,row)",col_err,",",row_err,")")

    # print("-- Predicted err (col,row): ({},{})".format(col_err[0],row_err[0]))

    # Calculate the reconstructed points on the non-shifted grid (in grid units).
    col_reco = col_err + arg_max[:,1] + 0.5
    row_reco = row_err + arg_max[:,0] + 0.5

    # Compute the 3x3 centroid.
    col_center = int((evt_arr.shape[2]-1)/2)
    row_center = int((evt_arr.shape[1]-1)/2)
    cmat_3x3 = torch.Tensor([[-1,0,1],[-1,0,1],[-1,0,1]]).cuda()
    rmat_3x3 = torch.Tensor([[-1,-1,-1],[0,0,0],[1,1,1]]).cuda()
    sum_3x3 = torch.sum(evt_arr[:,row_center-1:row_center+2,col_center-1:col_center+2],axis=(1,2))
    col_3x3 = torch.sum(cmat_3x3*evt_arr[:,row_center-1:row_center+2,col_center-1:col_center+2],axis=(1,2))/sum_3x3
    row_3x3 = torch.sum(rmat_3x3*evt_arr[:,row_center-1:row_center+2,col_center-1:col_center+2],axis=(1,2))/sum_3x3
    #print("3x3 CM in loss: (",col_3x3,",",row_3x3,")")

    # Calculate the distance between all pixel centers and the 3x3 centroid.
    dist_sq_3x3 = ((col_3x3 - col_err)**2 + (row_3x3 - row_err)**2)

    # Calculate the distance between all pixel centers and the reconstructed point.
    dist_reco = ((col_coords - col_err[:,None,None])**2 + (row_coords - row_err[:,None,None])**2)**0.5
    dist_reco_masked = dist_reco # temporary for outputting dist_reco while using 3x3 loss

    # # --------------------------------------------------------------------------
    # # Consider only distances for pixels with > 0.5*max_pixel_value.

    # # Get the maximum value in each pixel * 0.5.
    # # print("Evt array has shape",evt_arr.shape)
    # max_vals = torch.amax(evt_arr,axis=(1,2))*0.9 # (torch.amax(evt_arr,axis=(1,2))-8700)*0.5 + 8700
    # # print("Max vals first has shape",max_vals.shape)
    # max_vals = max_vals.unsqueeze(1).unsqueeze(1)
    # # print("Max vals with shape",max_vals.shape)

    # # Make a (frame_size,frame_size) matrix for each value.
    # max_mat = torch.ones(evt_arr.shape).cuda()
    # max_mat = max_vals*max_mat
    # # print("Max mat is",max_mat)
    # # print("Max mat with shape",max_mat.shape)

    # # Create the mask for each frame by comparing to each max_value*0.5 along the batch dimension.
    # mask_dist = (evt_arr > max_mat)
    # # print("dist_reco with shape",dist_reco.shape,"and mask with shape",mask_dist.shape)
    # dist_reco_masked = dist_reco*mask_dist              # apply the mask to zero the gradients in the non-masked elements
    # dist_reco_masked[~mask_dist] = torch.max(dist_reco)  # set all pixels outside max range to a large distance
    # # print("Dist reco masked",dist_reco_masked)
    # dist_reco_min = torch.amin(dist_reco_masked,axis=(1,2))
    # # print("Dist reco min after amin",dist_reco_min)
    # dist_reco_min = torch.max(dist_reco_min-0.5*2**0.5,torch.tensor(0.0))  # assume a tolerance of 0.5sqrt(2) pixels
    # # print("Dist reco",dist_reco)
    # # print("Dist reco min",dist_reco_min)

    # Compute the loss using the true point.
    # row_err_true = evt_err[:,1]/emnet.PIXEL_SIZE
    # col_err_true = evt_err[:,0]/emnet.PIXEL_SIZE
    # loss_vec = torch.mean((row_err_true - row_err)**2 + (col_err_true - col_err)**2)

    # Compute the loss term of mean((dist_reco_min)**2)
    #loss_vec = torch.mean((dist_reco_min)**2)
    # print("Max of dist_reco_min is",torch.max(dist_reco_min))
    # print("Min of dist_reco_min is",torch.min(dist_reco_min))
    # print("Vector loss is",loss_vec)

    # Compute the loss term using distance from the 3x3 point.
    loss_vec = torch.mean(dist_sq_3x3)

    # Compute the distance from the line for the reconstructed points.
    dist_line = (line_m*col_reco - row_reco + line_b) / (line_m**2 + 1)**0.5
    # print("Max of dist_line is",torch.max(dist_line))
    # print("Min of dist_line is",torch.min(dist_line))
    # print("arg_max",arg_max)
    # print("col_reco =",col_reco," and row_reco",row_reco)
    # print("Dist is",dist_line)

    # Below sign adjustment may change.
    #dist_line *= 1-2*light_region
    dist_line *= -(1-2*light_region)

    # Compute the loss term of (err)**2.
    #loss_vec = torch.mean(row_err**2 + col_err**2)

    # Compute the loss term concerning the distance from the line.
    loss_dist = torch.mean(torch.exp(-0.5*dist_line/sigma_dist))

    # Do not consider distance loss until the reconstruction error is small.
    if(torch.max(col_err,0).values > 11 or torch.max(row_err,0).values > 11):
        print("Not considering distance loss.")
        loss_dist = torch.tensor(0.0)

    # --------------------------------------------------------------------------
    # print("-- Vector loss: {}".format(loss_vec))
    # print("-- Distance loss: {}".format(loss_dist))

    # Weight the loss (if specified).
    #loss_weighted = torch.mean(wts*loss_total)
    #loss_weighted = torch.mean(loss_total) # no weights

    return loss_vec, loss_dist, dist_reco_masked

# Regression approach with line information
def train_regression_line(model, epoch, train_loader, optimizer, batch_size): # line_m, line_b

    # Compute row and col coordinates.
    indices = np.indices((emnet.EVT_SIZE,emnet.EVT_SIZE))
    row_coords = torch.tensor(indices[0] + 0.5 - ((emnet.EVT_SIZE-1)/2 + 0.5)).repeat([batch_size,1,1]).cuda()
    col_coords = torch.tensor(indices[1] + 0.5 - ((emnet.EVT_SIZE-1)/2 + 0.5)).repeat([batch_size,1,1]).cuda()

    losses_epoch = []; losses_vec_epoch = []; losses_dist_epoch = []; accuracies_epoch = []
    for batch_idx, (data, arg_max, evt_err, light_region, line_m, line_b) in enumerate(train_loader):

        data = data.cuda()
        data = Variable(data)
        arg_max = arg_max.cuda()
        evt_err = evt_err.cuda()
        light_region = light_region.cuda()
        line_m = line_m.cuda()
        line_b = line_b.cuda()

        optimizer.zero_grad()

        output_score = model(data)
        # print("Target shape is",target.shape)
        # print("Output score shape is",output_score.shape)

        loss_vec, loss_dist, _ = loss_reg_edge(data.squeeze(1),evt_err,output_score,row_coords,col_coords,arg_max,line_m,line_b,light_region)
        loss = loss_vec + loss_dist

        loss.backward()
        optimizer.step()

        # --------------------------------------------------------------------------
        # Determine the accuracy by computing the true and reconstructed points.
        row_err = evt_err[:,1]/emnet.PIXEL_SIZE
        col_err = evt_err[:,0]/emnet.PIXEL_SIZE
        row_true = row_err + arg_max[:,0] + 0.5
        col_true = col_err + arg_max[:,1] + 0.5
        row_out = output_score[:,1]
        col_out = output_score[:,0]
        row_reco = row_out + arg_max[:,0] + 0.5
        col_reco = col_out + arg_max[:,1] + 0.5

        correctvals = ((row_true - row_reco)**2 + (col_true - col_reco)**2)**0.5 < 0.1
        accuracy = correctvals.sum().float() / float(arg_max.size(0))
        # --------------------------------------------------------------------------

        # Get the current learning rate.
        param_group = optimizer.param_groups
        current_lr = param_group[0]['lr']

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t score_max: {:.6f}\t score_min: {:.6f}; Accuracy {:.3f}; LR {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), output_score.data.max(), output_score.data.min(), accuracy.data.item(), current_lr))

        losses_epoch.append(loss.data.item())
        losses_vec_epoch.append(loss_vec.data.item())
        losses_dist_epoch.append(loss_dist.data.item())
        accuracies_epoch.append(accuracy.data.item())

    print("---EPOCH AVG TRAIN LOSS:",np.mean(losses_epoch),"(VEC:",np.mean(losses_vec_epoch),", DIST:",np.mean(losses_dist_epoch),") ACCURACY:",np.mean(accuracies_epoch))
    with open("train.txt", "a") as ftrain:
        ftrain.write("{} {} {} {} {}\n".format(epoch,np.mean(losses_epoch),np.mean(losses_vec_epoch),np.mean(losses_dist_epoch),np.mean(accuracies_epoch)))

    return np.mean(losses_epoch)

# Regression approach
def train_regression(model, epoch, train_loader, optimizer):

    losses_epoch = []; accuracies_epoch = []
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output_score = model(data)
        # print("Target shape is",target.shape)
        # print("Output score shape is",output_score.shape)

        m = nn.MSELoss()
        loss = m(output_score,target)

        loss.backward()
        optimizer.step()

        #maxvals = output_score.argmax(dim=1)
        correctvals = (output_score - target)**2 < 0.0001
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

    return np.mean(losses_epoch)

def loss_edge(output, target, epoch = 0, sigma_dist = 1, w_edge = 100):
    output = output.squeeze(1)
    #print("target shape is",target.shape,"; output shape is",output.shape)

    th_truth = target[:,1,:,:]
    edge_truth = target[:,2,:,:]
    dist = target[:,3,:,:]

    # Define some Torch functions.
    sigmoid = torch.nn.Sigmoid()
    bce_loss = torch.nn.BCEWithLogitsLoss(reduce=False)

    # Modify the truth according to the epoch.
    # frac = min(epoch/500., 1.)
    # final_target = (1-frac)*th_truth + frac*sigmoid(output)

    final_target = th_truth

    # Compute the weights (tensor of shape [batchsize]).
    wts     = torch.sum(torch.exp(-(dist)**2/(2*sigma_dist**2))*th_truth,axis=(1,2))
    wt_norm = torch.sum(th_truth,axis=(1,2))
    wt_norm[wt_norm == 0] = 1
    wts /= wt_norm
    #wts[wts == 0] = 0.1

    # Zero-out the distance on the light side.
    dist_mod = torch.abs(dist*(edge_truth-1))

    # Compute the loss.
    #wts = torch.sum(torch.exp(-(dist)**2/(2*sigma_dist**2))*output,axis=0)
    #loss = torch.mean(torch.exp(-(dist)**2/(2*sigma_dist**2))*(bce_loss(output,th_truth) + w_edge*sigmoid(output)*(1-edge_truth)))
    #loss = torch.mean(torch.exp(-(dist)**2/(2*sigma_dist**2))*(bce_loss(output,final_target)))

    # --------------------------------------------------------------------------
    # BCE loss.
    loss_total = torch.sum(bce_loss(output,final_target),axis=(1,2))
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Constrained loss (tensor of shape [batchsize])
    # loss_bce = torch.sum(bce_loss(output,final_target),axis=(1,2))
    # loss_sum_constraint = torch.abs(torch.sum(sigmoid(output),axis=(1,2)) - 1)
    # loss_edge_penalty = w_edge*torch.sum(sigmoid(output)*dist_mod,axis=(1,2))
    #
    # #loss_total = loss_bce + loss_edge_penalty
    # loss_total = loss_edge_penalty + loss_sum_constraint
    # print("-- BCE loss: {}".format(loss_bce))
    # print("-- edge loss: {}".format(loss_edge_penalty))
    # print("-- sum-constraint loss: {}".format(loss_sum_constraint))
    # --------------------------------------------------------------------------

    # Weight the loss.
    #loss_weighted = torch.mean(wts*loss_total)
    loss_weighted = torch.mean(loss_total)
    return loss_weighted


def train_unet(model, epoch, train_loader, optimizer, sigma_dist = 2):

    losses_epoch = []; accuracies_epoch = []
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        #print("Target is",target)

        # Compute the final target.
        # final_target = th_truth * edge_truth
        # final_target = final_target.unsqueeze(1)

        output_score = model(data)
        #m = nn.BCEWithLogitsLoss(weight=wts)
        #loss = m(output_score,final_target)
        loss = loss_edge(output_score,target,epoch,w_edge = 1.0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()

        maxvals = (output_score[:,0,:,:] > 0.9)
        correctvals = (maxvals == target[:,0,:,:])
        accuracy = correctvals.sum().float() / float(target[:,0,:,:].nelement())

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t score_max: {:.6f}\t score_min: {:.6f}; Accuracy {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), output_score.data.max(), output_score.data.min(), accuracy.data.item()))

        losses_epoch.append(loss.data.item())
        accuracies_epoch.append(accuracy.data.item())

    print("---EPOCH AVG TRAIN LOSS:",np.mean(losses_epoch),"ACCURACY:",np.mean(accuracies_epoch))
    with open("train.txt", "a") as ftrain:
        ftrain.write("{} {} {}\n".format(epoch,np.mean(losses_epoch),np.mean(accuracies_epoch)))

    return np.mean(losses_epoch)

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

    return np.mean(losses_epoch)

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

    return np.mean(losses_epoch)


# ------------------------------------------------------------------------------
# OLD CODE

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
