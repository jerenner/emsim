"""
counting.py
"""
import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

from scipy.optimize import minimize
from scipy.stats import poisson

import torch
import torch.nn.functional as F

device = 'mps'

def gaussian_splash_pytorch(A, sigma, size=3, device='cpu'):
    """Prepare the Gaussian splash as a PyTorch convolution kernel."""
    x = y = torch.arange(0, size, device=device) - (size // 2)
    X, Y = torch.meshgrid(x, y)
    gaussian = A * torch.exp(- (X**2 + Y**2) / (2 * sigma**2))
    # Reshape to 4D tensor: (out_channels, in_channels, height, width)
    gaussian_kernel = gaussian.unsqueeze(0).unsqueeze(0)
    return gaussian_kernel

def construct_modeled_frame_pytorch(frame_ct, splash_kernel, device='cpu'):
    """Construct the modeled frame using convolution to apply the Gaussian splash."""
    # Ensure frame_ct is a float tensor and add batch and channel dimensions
    frame_ct_tensor = frame_ct.float().unsqueeze(1)
    
    # Apply the Gaussian splash across the entire frame using convolution
    modeled_frame = F.conv2d(frame_ct_tensor, splash_kernel, padding=splash_kernel.shape[-1]//2)
    
    # Add noise
    #noise_sigma = 1.0
    #modeled_frame = modeled_frame + torch.normal(0, noise_sigma, size=modeled_frame.shape, device=device)
    
    # Remove batch and channel dimensions from the output
    modeled_frame = modeled_frame.squeeze(1)
    
    return modeled_frame

def count_frame_pytorch(frame_bls, frame_ct, gauss_A, gauss_sigma, n_steps_max=500, loss_lim = 1):
    """Counts a frame given initial guess frame_ct"""
    
    # Convert frame and frame_ct to a PyTorch tensor
    frame_tensor = torch.from_numpy(frame_bls).to(device)
    frame_ct_tensor = torch.tensor(frame_ct, dtype=torch.float32, device=device, requires_grad=True)

    # Define the optimizer
    optimizer = torch.optim.Adam([frame_ct_tensor], lr=0.01)

    # Define the single-electron Gaussian splash
    splash = gaussian_splash_pytorch(gauss_A,gauss_sigma,device=device)

    # Set up the parameters for the iteration
    n_steps = 0
    loss = loss_lim + 1
    
    while(loss > loss_lim and n_steps < n_steps_max):
        
        optimizer.zero_grad()  # Clear previous gradients

        # Construct the modeled frame
        modeled_frame = construct_modeled_frame_pytorch(frame_ct_tensor, splash, device=device)

        # Compute the loss (negative likelihood)
        loss = torch.sum((frame_tensor - modeled_frame) ** 2)

        # Compute gradients
        loss.backward()

        # Update frame_ct_tensor based on gradients
        optimizer.step()

        n_steps += 1
        if(n_steps >= n_steps_max):
            print(f"Warning, stopping at max steps {n_steps} with loss {loss}")
        if n_steps % 100 == 0:
            print(f"Step {n_steps}, Loss: {loss.item()}")
    print(f"Counted in n_steps = {n_steps} with loss = {loss}")
    
    return frame_ct_tensor.cpu().detach().numpy(), modeled_frame.cpu().detach().numpy()

def frame_to_indices_weights(counted_frames):
    """Convert a batch of 2D counted frames into lists of linear indices and weights."""
    batch_size = counted_frames.shape[0]
    frame_shape = counted_frames.shape[1:]
    all_linear_indices = []
    all_weights = []

    for i in range(batch_size):
        frame = counted_frames[i]
        nonzero_indices = np.nonzero(frame)
        weights = frame[nonzero_indices]
        linear_indices = np.ravel_multi_index(nonzero_indices, frame_shape)
        
        all_linear_indices.append(linear_indices)
        all_weights.append(weights)

    return all_linear_indices, all_weights

def update_counted_data_hdf5(file_path, nframes, batch_start_idx, frames_indices, frames_weights, group_name='electron_events'):
    # Ensure frames_indices and frames_weights are lists of arrays, 
    # with each array corresponding to one frame's data.
    
    with h5py.File(file_path, 'a') as f:  # Open file in append mode
        # Create or access the group
        if group_name not in f:
            grp = f.create_group(group_name)
        else:
            grp = f[group_name]

        # Check if the VL datasets exist, create them if not
        if 'frames' not in grp:
            vl_dtype_indices = h5py.special_dtype(vlen=np.dtype('uint32'))
            vl_dataset_indices = grp.create_dataset("frames", (nframes,), dtype=vl_dtype_indices)
        else:
            vl_dataset_indices = grp['frames']

        if 'weights' not in grp:
            vl_dtype_weights = h5py.special_dtype(vlen=np.dtype('uint32'))
            vl_dataset_weights = grp.create_dataset("weights", (nframes,), dtype=vl_dtype_weights)
        else:
            vl_dataset_weights = grp['weights']

        # Assuming the length of frames_indices matches the expected number of frames,
        # iterate through each and update the datasets
        for i, (indices, weights) in enumerate(zip(frames_indices, frames_weights)):
            vl_dataset_indices[batch_start_idx+i] = indices
            vl_dataset_weights[batch_start_idx+i] = weights
            
def compute_conditional_probabilities(lam_grid):
    """
    Compute the array of conditional probabilities P(n >= 2) / P(n >= 1) for a grid of lambda values.
    
    :param lam_grid: A 576x576 numpy array of lambda values.
    :return: A 576x576 numpy array of conditional probabilities.
    """
    # Compute P(n >= 1) and P(n >= 2) for each lambda in the grid
    p_at_least_1 = 1 - poisson.cdf(0, lam_grid)  # P(n >= 1) = 1 - P(n < 1)
    p_at_least_2 = 1 - poisson.cdf(1, lam_grid)  # P(n >= 2) = 1 - P(n <= 1)
    
    # Conditional probability P(n >= 2) / P(n >= 1)
    # Safeguard division by zero by using np.where to only compute valid divisions
    conditional_prob = np.where(p_at_least_1 > 0, p_at_least_2 / p_at_least_1, 0)
    
    return conditional_prob

def compute_prior(frames_file, nframes, baseline, gauss_A):
    """
    Computes the prior using nframes from the specified file.

    :return: the prior, along with the conditional probabilities for having >= 2 counts
    """

    # Create the "prior"
    with h5py.File(frames_file, 'r') as f0:
        data = f0['frames']
        
        # Get all frames and subtract the baseline.
        prior_bls = np.array(data[0:nframes,:,:],dtype=np.float32) - baseline
        print(f"Prior values, {nframes} frames, shape: {prior_bls.shape}")
        
    # Compute the summed frame.
    prior_frame = np.sum(prior_bls,axis=0)
    print("Summed frame dimensions:",prior_frame.shape)

    # Divide by the average electron amplitude.
    prior_frame /= gauss_A

    # Normalize by the number of frames to get the final "prior".
    prior_frame /= nframes

    # Eliminate negative values.
    prior_frame[prior_frame < 0] = 0.

    return prior_frame

def count_frames(frames_file, counted_file, nframes, frame_width, frames_per_batch, th_single_elec, baseline, gauss_A, gauss_sigma, n_steps_max = 300, batch_start = 0, batch_end = -1, nframes_prior=0):
    """
    Counts the specified number of frames from the given file and saves the counted data to an HDF5 file.
    If nframes < 0, counts all frames in the file.
    """

    # Compute the prior if nframes_prior > 0.
    if(nframes_prior > 0):

        print(f"Computing the prior...")

        # Compute the prior.
        prior_frame = compute_prior(frames_file, nframes_prior, baseline, gauss_A)

        # Compute the conditional probabilities of having >= 2 counts, given >= 1 count.
        conditional_prob = compute_conditional_probabilities(prior_frame)

        # Repeat the conditional probabilities for the number of frames in a batch.
        conditional_prob_batch = np.repeat(conditional_prob[np.newaxis,:,:], frames_per_batch, axis=0)

    # Count all frames if nframes < 0.
    if(nframes < 0):
        with h5py.File(frames_file, 'r') as f0:
            data = f0['frames']
            nframes = data.shape[0]
            print(f"Counting all {nframes} frames")

    batches = round(nframes / frames_per_batch)
    print(f"Analyzing in {batches} batches...")
    if(batch_end < 0): batch_end = batches
    for batch in range(batches)[batch_start:batch_end]:
        print(f"\n\n ** BATCH {batch} **")
        
        # Get the frames for this batch
        print("-- Processing frames...")
        with h5py.File(frames_file, 'r') as f0:
            data = f0['frames']

            # Get the frames
            frame_bls = np.array(data[batch*frames_per_batch:batch*frames_per_batch+frames_per_batch,:,:],dtype=np.float32) - baseline

            # Compute an initial counted frame
            frame_ct = np.rint(frame_bls / gauss_A, out=np.zeros(frame_bls.shape,dtype=np.int16), casting='unsafe')
            frame_ct[frame_ct < th_single_elec] = 0

        # -------------------------------------------------------------------------------
        # Count the frames.
        print("-- Counting frames...")
        frame_ct_reco, modeled_frame_reco = count_frame_pytorch(frame_bls, frame_ct, gauss_A, gauss_sigma, n_steps_max = n_steps_max, loss_lim = nframes)
        frame_ct_reco[frame_ct_reco < 0] = 0
        frame_ct_reco = np.rint(frame_ct_reco)
        # -------------------------------------------------------------------------------
        
        # -------------------------------------------------------------------------------
        # Apply the prior for each frame.
        if(nframes_prior > 0):

            # Generate a 576x576 matrix of random numbers from a uniform distribution [0, 1)
            random_matrix = np.random.rand(*frame_ct_reco.shape)

            # Compare the random matrix to the probability matrix:
            #  - if the random number is greater than the conditional probability, 
            #    the multi-count is said to have been due to the Landau
            #    tail, and therefore the count is set to 1
            single_electron_pixels = random_matrix > conditional_prob_batch[:len(random_matrix)]

            # Set all counts > 1 that did not pass the probability game to 1 count.
            forced_single_electrons = (frame_ct_reco > 1) & single_electron_pixels
            n_single_elec = np.sum(forced_single_electrons)
            n_total = frames_per_batch*frame_width*frame_width
            print(f"{n_single_elec} of {n_total} ({n_single_elec/n_total*100:.4f}%) forced to single-electron counts")
            frame_ct_reco[forced_single_electrons] = 1
        # -------------------------------------------------------------------------------
        
        # Save the counted frames in the array
        print("-- Saving frames...")
        frames_indices, frames_weights = frame_to_indices_weights(frame_ct_reco)
        print(f"Frame indices len = {len(frames_indices)} and weights = {len(frames_weights)}")
        update_counted_data_hdf5(counted_file, nframes, batch*frames_per_batch, frames_indices, frames_weights)
