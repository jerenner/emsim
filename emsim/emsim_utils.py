import numpy as np
import pandas as pd

from emsim import emnet

from scipy.optimize import curve_fit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

rng_pixels = np.arange(-(emnet.EVT_SIZE-1)/2,(emnet.EVT_SIZE-1)/2+1)
coords_pixels_all = np.meshgrid(rng_pixels,rng_pixels)

def read_electron_data(fname, nevts=1000):

    evt = -1
    xinc = 0.0
    yinc = 0.0
    front = True
    energy = 0.0

    # Open the file and read the specified number of events.
    l_evt, l_xinc, l_yinc, l_front, l_energy, l_row, l_col, l_counts = [], [], [], [], [], [], [], []
    evts_read = 0
    with open(fname) as f:

        # Iterate through all lines.
        for line in f:

            # Stop reading if we've read the specified number of events.
            if(evts_read > nevts):
                break

            # Get each number in the line, separated by spaces.
            vals = line.rstrip().split(" ")

            # Start a new event.
            if(vals[0] == "EV"):
                evt    = vals[1]
                xinc   = vals[2]
                yinc   = vals[3]
                front  = (vals[4] == 1)
                energy = vals[5]
                evts_read += 1

            # Add a row for the current event.
            else:
                l_evt.append(int(evt))
                l_xinc.append(float(xinc))
                l_yinc.append(float(yinc))
                l_front.append(front)
                l_energy.append(float(energy))
                l_row.append(int(vals[0]))
                l_col.append(int(vals[1]))
                l_counts.append(int(vals[2]))

    # Construct the DataFrame.
    evt_dict = {'event': l_evt, 'xinc': l_xinc, 'yinc': l_yinc, 'front': l_front,  # x and y incidence points (true x and y points that led to the pixels; subpixel numbers)
                'energy': l_energy, 'row': l_row, 'col': l_col, 'counts': l_counts}
    df = pd.DataFrame.from_dict(evt_dict)

    return df

def compute_moments(evt_arr,args,order,pixel_size,coords_pixels):
    """
    Compute the x- and y-moments of the specified order.

    evt_arr: the original (unmodified) event array
    args:    the arguments into evt_arr of the pixels to be included
    order:   the order of the moment
    """

    xsum = 0  # x-moment sum
    ysum = 0  # y-moment sum
    tsum = 0  # total sum
    for arg in args:

        px = evt_arr[(arg[0],arg[1])]
        x  = coords_pixels[0][(arg[0],arg[1])]*pixel_size
        y  = coords_pixels[1][(arg[0],arg[1])]*pixel_size

        xsum += px*x**order
        ysum += px*y**order
        tsum += px

    if(tsum > 0):

        mx = xsum/tsum
        my = ysum/tsum

        return mx,my

    else:
        return 0.,0.


# ------------------------------------------------------------------------------
# Functions for computation of key quantities and evaluation on all events

# Event = single strike/snapshot
# centroid of pixel windows 3x3 and 5x5

def compute_key_quantities(evt_arr,threshold=40):
    """
    Get key quantities from the event array.
    """

    # Get the pixels above threshold and their corresponding arguments in the array.
    pixels_above_threshold = evt_arr[evt_arr > threshold]
    args_above_threshold = np.argwhere(evt_arr > threshold)
    arg_max = np.unravel_index(np.argmax(evt_arr),evt_arr.shape)
    xmax = coords_pixels_all[0][arg_max]*emnet.PIXEL_SIZE
    ymax = coords_pixels_all[1][arg_max]*emnet.PIXEL_SIZE
    #print("xmax",xmax,"ymax",ymax)

    # number of pixels above threshold
    n_above_threshold = len(args_above_threshold)

    # sum of pixels above threshold
    sum_above_threshold = np.sum(pixels_above_threshold)

    # maximum distance between pixels above threshold
    max_dist = 0
    for arg0 in args_above_threshold:
        for arg1 in args_above_threshold:
            dist = ((arg0[0] - arg1[0])**2 + (arg0[1] - arg1[1])**2)**0.5
            if(dist > max_dist):
                max_dist = dist
    max_dist_above_threshold = max_dist*emnet.PIXEL_SIZE

    # moments for pixels above threshold
    m1x, m1y = compute_moments(evt_arr,args_above_threshold,1,emnet.PIXEL_SIZE,coords_pixels_all)
    m2x, m2y = compute_moments(evt_arr,args_above_threshold,2,emnet.PIXEL_SIZE,coords_pixels_all)

    # moments for 3x3 region about maximum
    max_3x3 = evt_arr[max(arg_max[0]-1,0):min(arg_max[0]+2,evt_arr.shape[0]),max(arg_max[1]-1,0):min(arg_max[1]+2,evt_arr.shape[1])]
    coords_pixels_3x3 = [coords_pixels_all[0][max(arg_max[0]-1,0):min(arg_max[0]+2,evt_arr.shape[0]),max(arg_max[1]-1,0):min(arg_max[1]+2,evt_arr.shape[1])],
                         coords_pixels_all[1][max(arg_max[0]-1,0):min(arg_max[0]+2,evt_arr.shape[0]),max(arg_max[1]-1,0):min(arg_max[1]+2,evt_arr.shape[1])]]
    args_3x3 = np.argwhere(max_3x3 > 0)
    args_3x3_above_threshold = np.argwhere(max_3x3 > threshold)
    xc_3x3, yc_3x3 = compute_moments(max_3x3,args_3x3,1,emnet.PIXEL_SIZE,coords_pixels_3x3)
    xc_3x3_above_threshold, yc_3x3_above_threshold = compute_moments(max_3x3,args_3x3_above_threshold,1,emnet.PIXEL_SIZE,coords_pixels_3x3)

    xc_3x3 = xc_3x3
    yc_3x3 = yc_3x3
    xc_3x3_above_threshold = xc_3x3_above_threshold
    yc_3x3_above_threshold = yc_3x3_above_threshold

    # moments for 5x5 region about maximum
    max_5x5 = evt_arr[max(arg_max[0]-2,0):min(arg_max[0]+3,evt_arr.shape[0]),max(arg_max[1]-2,0):min(arg_max[1]+3,evt_arr.shape[1])]
    coords_pixels_5x5 = [coords_pixels_all[0][max(arg_max[0]-2,0):min(arg_max[0]+3,evt_arr.shape[0]),max(arg_max[1]-2,0):min(arg_max[1]+3,evt_arr.shape[1])],
                         coords_pixels_all[1][max(arg_max[0]-2,0):min(arg_max[0]+3,evt_arr.shape[0]),max(arg_max[1]-2,0):min(arg_max[1]+3,evt_arr.shape[1])]]
    args_5x5 = np.argwhere(max_5x5 > 0)
    args_5x5_above_threshold = np.argwhere(max_5x5 > threshold)
    xc_5x5, yc_5x5 = compute_moments(max_5x5,args_5x5,1,emnet.PIXEL_SIZE,coords_pixels_5x5)
    xc_5x5_above_threshold, yc_5x5_above_threshold = compute_moments(max_5x5,args_5x5_above_threshold,1,emnet.PIXEL_SIZE,coords_pixels_5x5)

    xc_5x5 = xc_5x5
    yc_5x5 = yc_5x5
    xc_5x5_above_threshold = xc_5x5_above_threshold
    yc_5x5_above_threshold = yc_5x5_above_threshold

    return [n_above_threshold, sum_above_threshold, max_dist_above_threshold,
            xmax, ymax,
            m1x, m1y, m2x, m2y,
            xc_3x3, yc_3x3, xc_3x3_above_threshold, yc_3x3_above_threshold,
            xc_5x5, yc_5x5, xc_5x5_above_threshold, yc_5x5_above_threshold]


def mult_gaussFun_Fit(x_y,*m):
    (x,y) = x_y
    A,x0,y0,varx,vary,C = m
    #print("x0 is {}".format(x0))
    X,Y = np.meshgrid(x,y)
    Z = A*np.exp(-0.5*((X-x0)**2/(varx)+(Y-y0)**2/(vary))) + C
    return Z.ravel()


# Compute sigma_x and sigma_y of the given probability distribution
def compute_sigmas(prob_dist, err_pixel_size, shifted_err_range_min):

    sum_tot = 0
    sum_x, sum_xsq = 0, 0
    sum_y, sum_ysq = 0, 0
    vmax = np.max(prob_dist)
    for i in range(prob_dist.size):

        xi = int(i % emnet.ERR_SIZE)*err_pixel_size + shifted_err_range_min + err_pixel_size/2
        yi = int(i / emnet.ERR_SIZE)*err_pixel_size + shifted_err_range_min + err_pixel_size/2

        vi = prob_dist[np.unravel_index(i,prob_dist.shape)]

        # Use a threshold of some fraction of vmax.
        if(vi > vmax / 10):
            sum_x   += xi*vi
            sum_xsq += xi**2*vi
            sum_y   += yi*vi
            sum_ysq += yi**2*vi
            sum_tot += vi

    # Compute mean and sigma.
    mean_x = sum_x/sum_tot
    mean_y = sum_y/sum_tot
    sigma_x = (sum_xsq/sum_tot - mean_x**2)**0.5
    sigma_y = (sum_ysq/sum_tot - mean_y**2)**0.5

    return sigma_x, sigma_y


def fit_sigmas(prob_dist,x,y,x0,y0,sigma_x0,sigma_y0,err_pixel_size):
    """
    Fit a 2D gaussian for sigma_x and sigma_y.

    prob_dist: the probability distribution
    x: a 1D list of the x-coordinates on the 2D grid (in mm)
    y: a 1D list of the y-coordinates on the 2D grid (in mm)
    x0: the initial guess for the x-value of the mean
    y0: the initial guess for the y-value of the mean
    sigma_x0: the initial guess for the sigma in the x-direction
    sigma_y0: the initial guess for the sigma in the y-direction
    """
    initial_guess = [np.max(prob_dist), x0, y0, sigma_x0**2, sigma_y0**2, np.max(prob_dist)/10.]
    bounds = ([0,x0-30*err_pixel_size,y0-30*err_pixel_size,0,0,0],[2*np.max(prob_dist),x0+30*err_pixel_size,y0+30*err_pixel_size,0.05,0.05,np.max(prob_dist)])

    try:
        popt, pcov = curve_fit(mult_gaussFun_Fit, (x, y), prob_dist.ravel(), p0=initial_guess, bounds=bounds)
    except (RuntimeError, ValueError):
        print("Error in fit; using initial guess")
        return initial_guess,None

    return popt, pcov


# makes
def construct_evt_dataframe(dset,evts,model,threshold=40):
    """
    Constructs a dataframe containing key information for each of the specified events.

    dset: the dataset of events
    evts: 1D array of event numbers to include in the dataframe
    model: the model to be used for making the prediction
    threshold: the threshold to be used in computation of relevant quantities

    """

    # Key quantities based on dataset shift.
    SHIFTED_ERR_RANGE_MIN = emnet.PIXEL_ERR_RANGE_MIN # - dset.add_shift*emnet.PIXEL_SIZE
    SHIFTED_ERR_RANGE_MAX = emnet.PIXEL_ERR_RANGE_MAX # + dset.add_shift*emnet.PIXEL_SIZE
    #ERR_PIXEL_SIZE = emnet.PIXEL_SIZE*(2*dset.add_shift+1)/emnet.ERR_SIZE
    ERR_PIXEL_SIZE = (emnet.PIXEL_ERR_RANGE_MAX - emnet.PIXEL_ERR_RANGE_MIN)/emnet.ERR_SIZE

    # Get the x and y coordinates of the 2D error prediction grid, in mm.
    x_errgrid = np.arange(0,emnet.ERR_SIZE)*ERR_PIXEL_SIZE + SHIFTED_ERR_RANGE_MIN + ERR_PIXEL_SIZE/2
    y_errgrid = np.arange(0,emnet.ERR_SIZE)*ERR_PIXEL_SIZE + SHIFTED_ERR_RANGE_MIN + ERR_PIXEL_SIZE/2

    # Create a softmax operation.
    softmax = nn.Softmax(dim=1)

    l_evt_arr = []
    l_evt, l_xtrue, l_ytrue, l_xpred, l_ypred, l_sigma_x_NN, l_sigma_y_NN = [], [], [], [], [], [], []

    l_n_above_threshold, l_sum_above_threshold, l_max_dist_above_threshold = [], [], []
    l_xmax, l_ymax = [], []
    l_m1_x_above_threshold, l_m1_y_above_threshold, l_m2_x_above_threshold, l_m2_y_above_threshold = [], [], [], []
    l_xc_3x3, l_yc_3x3, l_xc_3x3_above_threshold, l_yc_3x3_above_threshold = [], [], [], []
    l_xc_5x5, l_yc_5x5, l_xc_5x5_above_threshold, l_yc_5x5_above_threshold = [], [], [], []
    for evt in evts:

        evt_item = dset[evt]
        evt_arr = evt_item[0]
        evt_lbl = evt_item[1]
        evt_err_ind = evt_item[2]

        # Send through the model.
        data = torch.tensor(evt_arr).float().unsqueeze(0).unsqueeze(1).cuda()
        target = torch.tensor(np.array(evt_err_ind)).long().cuda()
        output_score = model(data)

        # Compute the predicted pixel and (x,y) values.
        prob = np.array(softmax(output_score).cpu().detach().numpy()).reshape([emnet.ERR_SIZE,emnet.ERR_SIZE])
        ipred = np.argmax(prob)
        xpred = int(ipred % emnet.ERR_SIZE)*ERR_PIXEL_SIZE + SHIFTED_ERR_RANGE_MIN + ERR_PIXEL_SIZE/2
        ypred = int(ipred / emnet.ERR_SIZE)*ERR_PIXEL_SIZE + SHIFTED_ERR_RANGE_MIN + ERR_PIXEL_SIZE/2
        #print("[Evt",evt,"]: Index is",evt_err_ind,"with predicted",ipred,"; x = {} (predicted {}), y = {} (predicted {})".format(evt_lbl[0],xpred,evt_lbl[1],ypred))

        # Compute the sigmas of the distribution.
        sigma_x0_NN, sigma_y0_NN = compute_sigmas(prob,ERR_PIXEL_SIZE,SHIFTED_ERR_RANGE_MIN)
        popt, pcov = fit_sigmas(prob,x_errgrid,y_errgrid,xpred,ypred,sigma_x0_NN,sigma_y0_NN,ERR_PIXEL_SIZE)
        #xpred = popt[1]
        #ypred = popt[2]
        sigma_x_NN = popt[3]**0.5
        sigma_y_NN = popt[4]**0.5

        [n_above_threshold, sum_above_threshold, max_dist_above_threshold,
            xmax, ymax,
            m1x_above_threshold, m1y_above_threshold, m2x_above_threshold, m2y_above_threshold,
            xc_3x3, yc_3x3, xc_3x3_above_threshold, yc_3x3_above_threshold,
            xc_5x5, yc_5x5, xc_5x5_above_threshold, yc_5x5_above_threshold] = compute_key_quantities(evt_arr)

        # Fill the lists.
        l_evt.append(evt)
        l_xtrue.append(evt_lbl[0])
        l_ytrue.append(evt_lbl[1])
        l_xpred.append(xpred)
        l_ypred.append(ypred)
        l_sigma_x_NN.append(sigma_x_NN)
        l_sigma_y_NN.append(sigma_y_NN)

        l_evt_arr.append(evt_arr)

        l_n_above_threshold.append(n_above_threshold)
        l_sum_above_threshold.append(sum_above_threshold)
        l_max_dist_above_threshold.append(max_dist_above_threshold)
        l_xmax.append(xmax)
        l_ymax.append(ymax)
        l_m1_x_above_threshold.append(m1x_above_threshold)
        l_m1_y_above_threshold.append(m1y_above_threshold)
        l_m2_x_above_threshold.append(m2x_above_threshold)
        l_m2_y_above_threshold.append(m2y_above_threshold)
        l_xc_3x3.append(xc_3x3)
        l_yc_3x3.append(yc_3x3)
        l_xc_3x3_above_threshold.append(xc_3x3_above_threshold)
        l_yc_3x3_above_threshold.append(yc_3x3_above_threshold)
        l_xc_5x5.append(xc_5x5)
        l_yc_5x5.append(yc_5x5)
        l_xc_5x5_above_threshold.append(xc_5x5_above_threshold)
        l_yc_5x5_above_threshold.append(yc_5x5_above_threshold)

        if((evt-evts[0]) % (len(evts)/100) == 0):
            print("{}% done".format(int((evt-evts[0]) / (len(evts)/100))))

    # Create the dataframe.
    evt_dict = {'event': l_evt, 'x_true': l_xtrue, 'y_true': l_ytrue, 'x_pred': l_xpred, 'y_pred': l_ypred,
                'sigma_x_NN': l_sigma_x_NN, 'sigma_y_NN': l_sigma_y_NN,
                'n_above_threshold': l_n_above_threshold,
                'sum_above_threshold': l_sum_above_threshold,
                'max_dist_above_threshold': l_max_dist_above_threshold,
                'xmax_pixel': l_xmax,
                'ymax_pixel': l_ymax,
                'm1_x_above_threshold': l_m1_x_above_threshold,
                'm1_y_above_threshold': l_m1_y_above_threshold,
                'm2_x_above_threshold': l_m2_x_above_threshold,
                'm2_y_above_threshold': l_m2_y_above_threshold,
                'xc_3x3': l_xc_3x3,
                'yc_3x3': l_yc_3x3,
                'xc_3x3_above_threshold': l_xc_3x3_above_threshold,
                'yc_3x3_above_threshold': l_yc_3x3_above_threshold,
                'xc_5x5': l_xc_5x5,
                'yc_5x5': l_yc_5x5,
                'xc_5x5_above_threshold': l_xc_5x5_above_threshold,
                'yc_5x5_above_threshold': l_yc_5x5_above_threshold}
    df = pd.DataFrame.from_dict(evt_dict)

    # Create derived quantities.
    df["sigma_r_NN"] = (df.sigma_x_NN**2 + df.sigma_y_NN**2)**0.5
    df["error_x_NN"] = df.x_pred - df.x_true
    df["error_y_NN"] = df.y_pred - df.y_true
    df["error_r_NN"] = (df.error_x_NN**2 + df.error_y_NN**2)**0.5
    df["sigma_x_above_threshold"] = df.m2_x_above_threshold - df.m1_x_above_threshold**2
    df["sigma_y_above_threshold"] = df.m2_y_above_threshold - df.m1_y_above_threshold**2
    df["sigma_max_above_threshold"] = df[["sigma_x_above_threshold", "sigma_y_above_threshold"]].max(axis=1)
    df["sigma_min_above_threshold"] = df[["sigma_x_above_threshold", "sigma_y_above_threshold"]].min(axis=1)
    df["error_x_maxpt"] = df.xmax_pixel - df.x_true
    df["error_y_maxpt"] = df.ymax_pixel - df.y_true
    df["error_r_maxpt"] = (df.error_x_maxpt**2 + df.error_y_maxpt**2)**0.5
    df["error_x_th"] = df.m1_x_above_threshold - df.x_true
    df["error_y_th"] = df.m1_y_above_threshold - df.y_true
    df["error_r_th"] = (df.error_x_th**2 + df.error_y_th**2)**0.5
    df["error_x_3x3"] = df.xc_3x3 - df.x_true
    df["error_y_3x3"] = df.yc_3x3 - df.y_true
    df["error_r_3x3"] = (df.error_x_3x3**2 + df.error_y_3x3**2)**0.5
    df["error_x_3x3_th"] = df.xc_3x3_above_threshold - df.x_true
    df["error_y_3x3_th"] = df.yc_3x3_above_threshold - df.y_true
    df["error_r_3x3_th"] = (df.error_x_3x3_th**2 + df.error_y_3x3_th**2)**0.5
    df["error_x_5x5"] = df.xc_5x5 - df.x_true
    df["error_y_5x5"] = df.yc_5x5 - df.y_true
    df["error_r_5x5"] = (df.error_x_5x5**2 + df.error_y_5x5**2)**0.5
    df["error_x_5x5_th"] = df.xc_5x5_above_threshold - df.x_true
    df["error_y_5x5_th"] = df.yc_5x5_above_threshold - df.y_true
    df["error_r_5x5_th"] = (df.error_x_5x5_th**2 + df.error_y_5x5_th**2)**0.5

    return df, np.array(l_evt_arr)
